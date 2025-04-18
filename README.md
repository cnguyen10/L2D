[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

The implementation in this branch is to mimic the method LECODU proposed in the paper [Learning to Complement and to Defer to Multiple Users](https://link.springer.com/chapter/10.1007/978-3-031-72992-8_9).

The difference is that there is no specific human identities. Hence, the gating model will defer to either one, two and up to all human experts, where each of them is randomly selected.