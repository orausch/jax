#!/bin/bash

# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# More or less copied from
# https://github.com/iree-org/iree/tree/main/build_tools/github_actions/runner/config

set -ex

# Setup pre-job hook
cat ~/jax/.github/workflows/self_hosted_runner_utils/runner.env | envsubst >> ~/actions-runner/.env

# Setup Github Actions Runner to automatically start on reboot (e.g. due to TPU
# VM maintenance events)
echo "@reboot $HOME/jax/.github/workflows/self_hosted_runner_utils/start_github_runner.sh" | crontab -
