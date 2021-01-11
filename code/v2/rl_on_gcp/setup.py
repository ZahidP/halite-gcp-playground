# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Source: https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-rl-pong/setup.py"""

from setuptools import find_packages
from setuptools import setup

from distutils.command.build import build as _build


class build(_build):
    """A build command class that will be invoked during package install.

      The package built using the current setup.py will be staged and later
      installed in the worker using `pip install package'. This class will be
      instantiated during install for this specific scenario and will trigger
      running the custom commands specified.
      """
    sub_commands = _build.sub_commands + [('CustomCommands', None)]


#####

REQUIRED_PACKAGES = [
    'numpy', 'agents==1.4.0', 'tensorflow==2.2.0', 'opencv_python==3.4.3.18', 'kaggle-environments==1.7.3'
]

setup(
    name='rl_on_gcp',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description='Deep Q for Halite.',
)
