import os
from pathlib import Path
from functools import partial

from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

import jax
import jax.numpy as jnp

from flax import nnx
from flax.traverse_util import flatten_dict

import optax

import orbax.checkpoint as ocp

from jaxopt import ProjectedGradient
from jaxopt.projection import projection_non_negative

import mlflow

import grain.python as grain

from DataSource import ImageDataSource
from transformations import (
    RandomCrop,
    Resize,
    RandomHorizontalFlip,
    Normalize,
    ToFloat
)


def init_tx(dataset_length: int, cfg: DictConfig) -> optax.GradientTransformationExtraArgs:
    """initialize parameters of an optimizer
    """
     # add L2 regularisation(aka weight decay)
    weight_decay = optax.masked(
        inner=optax.add_decayed_weights(
            weight_decay=cfg.training.weight_decay,
            mask=None
        ),
        mask=lambda p: jax.tree_util.tree_map(lambda x: x.ndim != 1, p)
    )

    num_iters_per_epoch = dataset_length // cfg.training.batch_size
    lr_schedule_fn = optax.cosine_decay_schedule(
        init_value=cfg.training.lr,
        decay_steps=(cfg.training.num_epochs + 10) * num_iters_per_epoch
    )

    # define an optimizer
    tx = optax.chain(
        weight_decay,
        optax.add_noise(eta=0.01, gamma=0.55, seed=cfg.training.seed),
        optax.clip_by_global_norm(max_norm=cfg.training.clipped_norm) \
            if cfg.training.clipped_norm is not None else optax.identity(),
        optax.sgd(learning_rate=lr_schedule_fn, momentum=cfg.training.momentum)
    )

    return tx


def initialize_dataloader(
    data_source: grain.RandomAccessDataSource,
    num_epochs: int,
    shuffle: bool,
    seed: int,
    batch_size: int,
    crop_size: tuple[int, int] = None,
    resize: tuple[int, int] = None,
    mean: float = None,
    std: float = None,
    prob_random_h_flip: float = None,
    num_workers: int = 0,
    num_threads: int = 1,
    prefetch_size: int = 1
) -> grain.IterDataset:
    """
    """
    index_sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=num_epochs,
        shuffle=shuffle,
        shard_options=grain.NoSharding(),
        seed=seed  # set the random seed
    )

    transformations = []

    if resize is not None:
        transformations.append(Resize(resize_shape=resize))

    if crop_size is not None:
        transformations.append(RandomCrop(crop_size=crop_size))

    transformations.append(RandomHorizontalFlip(p=prob_random_h_flip))
    transformations.append(ToFloat())

    if mean is not None and std is not None:
        transformations.append(Normalize(mean=mean, std=std))

    transformations.append(
        grain.Batch(
            batch_size=batch_size,
            drop_remainder=shuffle
        )
    )

    data_loader = grain.DataLoader(
        data_source=data_source,
        sampler=index_sampler,
        operations=transformations,
        worker_count=num_workers,
        shard_options=grain.NoSharding(),
        read_options=grain.ReadOptions(
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_size
        )
    )

    return iter(data_loader)


@jax.jit
def get_unnorm_log_q_z_tilde(q_uncon: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    """calculate the un-normalised q(z)

    Args:
        q_uncon: the posterior of z without any constraints
        params: a tree-like or dictionary containing the Lagrange multipliers
            - upper:
            - lower:

    Return:
        log_q: the logarithm of the un-normalised posterior
    """
    # calculate log_q_tilde
    log_q_den = params['upper'] - params['lower'] + 1
    log_q = jnp.log(q_uncon) - log_q_den  # (batch, num_experts + 1)

    return log_q


@jax.jit
def constrained_posterior(
    q_z_uncon: jax.Array,
    epsilon_upper: jax.Array,
    epsilon_lower: jax.Array
) -> jax.Array:
    """calculate the work-load balancing posterior

    Args:
        log_q_z_unconstrained: the unconstrained posterior of z
        epsilon_upper: the hyperparameter for upper constraint
        epsilon_lower: the hyperparameter for lower constraint

    Returns:
        log_q_z: the constrained posterior of z
    """
    def duality_Lagrangian(params: dict[str, jax.Array]) -> jax.Array:
        """the duality of Lagrangian to find the Lagrange multiplier

        Args:
            lmbd: a dictionary containing the Lagrange multiplier with the following keys:
                - upper: corresponds to epsilon upper  # (num_experts + 1,)
                - lower: epsilon lower  # (num_experts + 1,)
                - ij: epsilon ij  # (batch, num_experts + 1)

        Returns:
            lagrangian:
        """
        log_q_tilde = get_unnorm_log_q_z_tilde(q_uncon=q_z_uncon, params=params)

        # calculate Lagrangian
        lgr = jax.nn.logsumexp(a=log_q_tilde, axis=-1)
        lgr = jnp.mean(a=lgr, axis=0)

        lgr = lgr + jnp.sum(a=params['upper'] * epsilon_upper, axis=0)
        lgr = lgr - jnp.sum(a=params['lower'] * epsilon_lower, axis=0)

        return lgr

    init_params = dict(
        upper=jnp.zeros_like(a=epsilon_upper, dtype=jnp.float32),
        lower=jnp.zeros_like(a=epsilon_lower, dtype=jnp.float32)
    )

    pg = ProjectedGradient(fun=duality_Lagrangian, projection=projection_non_negative)
    res = pg.run(init_params=init_params)

    log_q_z = get_unnorm_log_q_z_tilde(q_uncon=q_z_uncon, params=res.params)

    # normalisation
    log_q_z -= jax.nn.logsumexp(a=log_q_z, axis=-1, keepdims=True)
    q_z = jnp.exp(log_q_z)

    return q_z


@nnx.jit
def unconstrained_posterior(
    gating_model: nnx.Module,
    x: jax.Array,
    t: jax.Array,
    y: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    """
    # CALCULATE Pr(y | z, t)
    log_p_y_zt = y[:, None, :] * jnp.log(t)  # (batch, num_experts, num_classes)
    log_p_y_zt = jnp.sum(a=log_p_y_zt, axis=-1)  # (batch, num_experts)

    # P(z | x, gamma)
    logit_p_z_x_gamma = gating_model(x)  # (batch, num_experts)
    log_p_z_x_gamma = jax.nn.log_softmax(x=logit_p_z_x_gamma, axis=-1)

    # Pr(z | x, y, t)
    log_p_z_xyt = log_p_y_zt + log_p_z_x_gamma  # (batch, num_experts)
    log_p_z_xyt = log_p_z_xyt - jax.nn.logsumexp(a=log_p_z_xyt, axis=-1, keepdims=True)
    p_z_xyt = jnp.exp(log_p_z_xyt)

    return p_z_xyt, logit_p_z_x_gamma


def expectation_step(
    gating_model: nnx.Module,
    x: jax.Array,
    t: jax.Array,
    y: jax.Array,
    epsilon_upper: jax.Array,
    epsilon_lower: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    """
    p_z_xyt, logit_p_z_x_gamma = unconstrained_posterior(
        gating_model=gating_model,
        x=x,
        t=t,
        y=y
    )

    p_z_xyt = jax.lax.stop_gradient(x=p_z_xyt)

    q_z = constrained_posterior(p_z_xyt, epsilon_upper, epsilon_lower)

    return q_z, logit_p_z_x_gamma


def variational_free_energy(
    gating_model: nnx.Module,
    x: jax.Array,
    t: jax.Array,
    y: jax.Array,
    epsilon_upper: jax.Array,
    epsilon_lower: jax.Array
) -> jax.Array:
    """
    """
    q_z, logit_p_z_x_gamma = expectation_step(
        gating_model=gating_model,
        x=x,
        t=t,
        y=y,
        epsilon_upper=epsilon_upper,
        epsilon_lower=epsilon_lower
    )

    # loss as the variational-free energy
    loss = optax.losses.softmax_cross_entropy(logits=logit_p_z_x_gamma, labels=q_z)
    loss = jnp.mean(a=loss, axis=0)

    return loss


def classification_loss(
    model: nnx.Module,
    x: jax.Array,
    y: jax.Array
) -> jax.Array:
    """classification loss to train a classifier on ground truth data
    """
    logits = model(x)
    loss = optax.losses.softmax_cross_entropy(
        logits=logits,
        labels=y
    )  # (batch,)
    loss = jnp.mean(a=loss, axis=0)

    return loss


@partial(nnx.jit, static_argnames=('cfg',), donate_argnames=('gating', 'clf'))
def expectation_maximisation(
    x: jax.Array,
    t: jax.Array,
    y: jax.Array,
    gating: nnx.Optimizer,
    clf: nnx.Optimizer,
    cfg: DictConfig
) -> tuple[nnx.Optimizer, jax.Array, dict[str, jax.Array]]:
    """perform the variational EM
    """
    t = jax.nn.one_hot(x=t, num_classes=cfg.dataset.num_classes)  # (batch, num_experts, num_classes)
    t = optax.smooth_labels(labels=t, alpha=0.01)

    y = jax.nn.one_hot(x=y, num_classes=cfg.dataset.num_classes)  # (batch, num_classes)

    epsilon_upper = jnp.array(object=cfg.hparams.epsilon_upper)
    epsilon_lower = jnp.array(object=cfg.hparams.epsilon_lower)

    # prediction of classifier
    clf.model.eval()
    logits_clf = jax.lax.stop_gradient(x=clf.model(x))
    clf.model.train()
    p_clf = jax.nn.softmax(x=logits_clf, axis=-1)  # (batch, num_classes)

    # concatenate to annotations
    t = jnp.concatenate(arrays=(t, p_clf[:, None, :]), axis=1)  # (batch, num_experts + 1, num_classes)

    # gating
    grad_fn_gating = nnx.value_and_grad(f=variational_free_energy, argnums=0)
    loss_gating, grads_gating = grad_fn_gating(
        gating.model,
        x,
        t,
        y,
        epsilon_upper,
        epsilon_lower
    )

    # in-place update
    gating.update(grads=grads_gating)

    # classifier
    grad_fn_clf = nnx.value_and_grad(f=classification_loss, argnums=0)
    loss_clf, grads_clf = grad_fn_clf(model=clf.model, x=x, y=y)
    clf.update(grads=grads_clf)

    return gating, clf, {'loss/gating': loss_gating, 'loss/clf': loss_clf}


def train(
    dataloader: grain.DatasetIterator,
    gating: nnx.Optimizer,
    clf: nnx.Optimizer,
    cfg: DictConfig
) -> tuple[nnx.Optimizer, nnx.Optimizer, dict[str, jax.Array]]:
    """
    """
    loss_metrics_dict = {
        'loss/gating': nnx.metrics.Average(),
        'loss/clf': nnx.metrics.Average()
    }

    gating.model.train()
    clf.model.train()

    for _ in tqdm(
        iterable=range(cfg.dataset.length.train // cfg.training.batch_size),
        desc='train',
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg.data_loading.progress_bar
    ):
        samples = next(dataloader)
        x = jnp.asarray(a=samples['image'], dtype=jnp.float32)  # input samples
        t = jnp.asarray(a=samples['label'], dtype=jnp.int32)  # annotated labels (batch, num_experts)
        y = jnp.asarray(a=samples['ground_truth'], dtype=jnp.int32)  # (batch,)

        gating, clf, loss_dict = expectation_maximisation(
            x=x,
            t=t,
            y=y,
            gating=gating,
            clf=clf,
            cfg=cfg
        )

        if jnp.isnan(loss_dict['loss/gating']):
            raise ValueError('Training loss is NaN.')

        for key in loss_dict:
            loss_metrics_dict[key].update(values=loss_dict[key])

    for key in loss_metrics_dict:
        loss_metrics_dict[key] = loss_metrics_dict[key].compute()

    return (gating, clf, loss_metrics_dict)


def evaluation(
    dataloader: grain.DatasetIterator,
    gating: nnx.Module,
    clf: nnx.Module,
    cfg: DictConfig
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    """
    # set evaluation mode
    gating.eval()
    clf.eval()

    accuracy_accum = nnx.metrics.Accuracy()
    clf_accuracy = nnx.metrics.Accuracy()
    coverage = nnx.metrics.Average()

    for _ in tqdm(
        iterable=range(cfg.dataset.length.test // cfg.training.batch_size),
        desc='eval',
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg.data_loading.progress_bar
    ):
        samples = next(dataloader)
        x = jnp.asarray(a=samples['image'], dtype=jnp.float32)  # input samples
        y = jnp.asarray(a=samples['ground_truth'], dtype=jnp.int32)  # true labels (batch_size,)
        t = jnp.asarray(a=samples['label'], dtype=jnp.int32)  # annotated labels (batch_size, num_experts)

        t = jax.nn.one_hot(x=t, num_classes=cfg.dataset.num_classes)  # (batch_size, num_experts, num_classes)

        # Pr(z | x, gamma)
        logits_p_z = gating(x)  # (batch_size, num_experts)
        if jnp.isnan(logits_p_z).any():
            raise ValueError('NaN detected in the output of gating function.')

        selected_expert_ids = jnp.argmax(a=logits_p_z, axis=-1)  # (batch_size,)

        coverage.update(values=(selected_expert_ids == len(cfg.dataset.test_files)) * 1)

        # accuracy
        logits_clf = clf(x)  # (batch_size, num_classes)
        human_and_model_predictions = jnp.concatenate(arrays=(t, logits_clf[:, None, :]), axis=1)
        queried_predictions = human_and_model_predictions[jnp.arange(len(x)), selected_expert_ids, :]
        accuracy_accum.update(logits=queried_predictions, labels=y)

        # classifier accuracy
        clf_accuracy.update(logits=logits_clf, labels=y)

    return (accuracy_accum.compute(), clf_accuracy.compute(), coverage.compute())


@hydra.main(version_base=None, config_path="./conf", config_name="conf")
def main(cfg: DictConfig) -> None:
    """
    """
    # region JAX ENVIRONMENT
    jax.config.update('jax_disable_jit', cfg.jax.disable_jit)
    jax.config.update('jax_platforms', cfg.jax.platform)

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(cfg.jax.mem)
    # endregion

    # region DATASETS
    datasource_train = ImageDataSource(
        annotation_files=cfg.dataset.train_files,
        ground_truth_file=cfg.dataset.train_ground_truth_file,
        root=cfg.dataset.root,
        num_samples=cfg.training.num_samples,
        seed=cfg.training.seed
    )
    datasource_test = ImageDataSource(
        annotation_files=cfg.dataset.test_files,
        ground_truth_file=cfg.dataset.test_ground_truth_file,
        root=cfg.dataset.root
    )

    OmegaConf.set_struct(conf=cfg, value=True)
    OmegaConf.update(
        cfg=cfg,
        key='dataset.length.train',
        value=len(datasource_train),
        force_add=True
    )
    OmegaConf.update(
        cfg=cfg,
        key='dataset.length.test',
        value=len(datasource_test),
        force_add=True
    )
    # endregion

    # region MODELS
    model_fn = hydra.utils.instantiate(config=cfg.model)  # function to instanitate

    gating = nnx.Optimizer(
        model=model_fn(
            num_classes=len(cfg.dataset.train_files) + 1,
            rngs=nnx.Rngs(jax.random.key(seed=cfg.training.seed)),
            dropout_rate=cfg.training.dropout_rate,
            dtype=eval(cfg.jax.dtype)
        ),
        tx=init_tx(dataset_length=len(datasource_train), cfg=cfg)
    )

    clf = nnx.Optimizer(
        model=model_fn(
            num_classes=cfg.dataset.num_classes,
            rngs=nnx.Rngs(jax.random.key(seed=cfg.training.seed)),
            dropout_rate=cfg.training.dropout_rate,
            dtype=eval(cfg.jax.dtype)
        ),
        tx=init_tx(dataset_length=len(datasource_train), cfg=cfg)
    )

    # options to store models
    ckpt_options = ocp.CheckpointManagerOptions(
        save_interval_steps=100,
        max_to_keep=1,
        step_format_fixed_length=3,
        enable_async_checkpointing=True
    )
    # endregion

    mlflow.set_tracking_uri(uri=cfg.experiment.tracking_uri)
    mlflow.set_experiment(experiment_name=cfg.experiment.name)
    mlflow.disable_system_metrics_logging()
    # mlflow.set_system_metrics_sampling_interval(interval=600)
    # mlflow.set_system_metrics_samples_before_logging(samples=1)

    # create a directory for storage (if not existed)
    if not os.path.exists(path=cfg.experiment.logdir):
        Path(cfg.experiment.logdir).mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(
        run_id=cfg.experiment.run_id,
        log_system_metrics=False
    ) as mlflow_run:
        # append run id into the artifact path
        ckpt_dir = os.path.join(
            os.getcwd(),
            cfg.experiment.logdir,
            cfg.experiment.name,
            mlflow_run.info.run_id
        )

        # enable an orbax checkpoint manager to save model's parameters
        with ocp.CheckpointManager(
            directory=ckpt_dir,
            options=ckpt_options,
            item_names=('gating', 'clf')
        ) as ckpt_mngr:
            # region LOGGING and RESTORING
            if cfg.experiment.run_id is None:
                start_epoch_id = 0

                # log hyper-parameters
                mlflow.log_params(
                    params=flatten_dict(xs=OmegaConf.to_container(cfg=cfg), sep='.')
                )

                # log source code
                mlflow.log_artifact(
                    local_path=os.path.abspath(path=__file__),
                    artifact_path='source_code'
                )
            else:
                start_epoch_id = ckpt_mngr.latest_step()

                checkpoint = ckpt_mngr.restore(
                    step=start_epoch_id,
                    args=ocp.args.Composite(
                        gating=ocp.args.StandardRestore(item=nnx.state(node=gating.model)),
                        clf=ocp.args.StandardRestore(item=nnx.state(node=clf.model))
                    )
                )

                nnx.update(gating.model, checkpoint.gating)
                nnx.update(clf.model, checkpoint.clf)

                del checkpoint
            # endregion

            # region DATA LOADERS
            dataloader_train = initialize_dataloader(
                data_source=datasource_train,
                num_epochs=cfg.training.num_epochs - start_epoch_id,
                shuffle=True,
                seed=cfg.training.seed,
                batch_size=cfg.training.batch_size,
                crop_size=cfg.hparams.crop_size,
                resize=cfg.hparams.resize,
                mean=cfg.hparams.mean,
                std=cfg.hparams.std,
                prob_random_h_flip=cfg.hparams.prob_random_h_flip,
                num_workers=cfg.data_loading.num_workers,
                num_threads=cfg.data_loading.num_threads,
                prefetch_size=cfg.data_loading.prefetch_size
            )

            dataloader_test = initialize_dataloader(
                data_source=datasource_test,
                num_epochs=cfg.training.num_epochs - start_epoch_id,
                shuffle=False,
                seed=0,
                batch_size=cfg.training.batch_size,
                crop_size=cfg.hparams.crop_size,
                resize=cfg.hparams.resize,
                mean=cfg.hparams.mean,
                std=cfg.hparams.std,
                prob_random_h_flip=cfg.hparams.prob_random_h_flip,
                num_workers=cfg.data_loading.num_workers,
                num_threads=cfg.data_loading.num_threads,
                prefetch_size=cfg.data_loading.prefetch_size
            )
            # endregion

            for epoch_id in tqdm(
                iterable=range(start_epoch_id, cfg.training.num_epochs, 1),
                desc='progress',
                ncols=80,
                leave=True,
                position=1,
                colour='green',
                disable=not cfg.data_loading.progress_bar
            ):
                # training
                gating, clf, loss_dict = train(
                    dataloader=dataloader_train,
                    gating=gating,
                    clf=clf,
                    cfg=cfg
                )

                mlflow.log_metrics(
                    metrics=loss_dict,
                    step=epoch_id + 1,
                    synchronous=False
                )

                if (epoch_id + 1) % cfg.training.eval_every_n_epochs == 0:
                    # evaluation
                    accuracy, clf_accuracy, coverage = evaluation(
                        dataloader=dataloader_test,
                        gating=gating.model,
                        clf=clf.model,
                        cfg=cfg
                    )

                    mlflow.log_metrics(
                        metrics={
                            'accuracy/l2d': accuracy,
                            'accuracy/clf': clf_accuracy,
                            'coverage/clf': coverage
                        },
                        step=epoch_id + 1,
                        synchronous=False
                    )

                # wait until completing the asynchronous saving
                ckpt_mngr.wait_until_finished()

                # save parameters asynchronously
                ckpt_mngr.save(
                    step=epoch_id + 1,
                    args=ocp.args.Composite(
                        gating=ocp.args.StandardSave(nnx.state(node=gating.model)),
                        clf=ocp.args.StandardSave(nnx.state(node=clf.model))
                    )
                )

    return None


if __name__ == '__main__':
    # cache Jax compilation to reduce compilation time in next runs
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 120)

    main()
