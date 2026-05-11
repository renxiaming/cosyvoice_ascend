# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse

from components.utils.parser import BaseCommand, AitInstallCommand, AitBuildExtraCommand, \
                                    AitCheckCommand, DownloadCommand
from components.utils.constants import AIT_FAQ_HOME, MIND_STUDIO_LOGO
from components.utils.log import logger


class UmaskWrapper:
    """Write with preset umask
    >>> with UmaskWrapper():
    >>>     ...
    """

    def __init__(self, umask=0o027):
        self.umask, self.ori_umask = umask, None

    def __enter__(self):
        self.ori_umask = os.umask(self.umask)

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        os.umask(self.ori_umask)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"msit(MindStudio Inference Tools), {MIND_STUDIO_LOGO}.\n"
        "Providing one-site debugging and optimization toolkit for inference on Ascend Devices.\n"
        f"For any issue, refer FAQ first: {AIT_FAQ_HOME}",
    )

    cmd = BaseCommand(
        "msit", None, ["msit_sub_task", AitInstallCommand(), AitBuildExtraCommand(), 
                       AitCheckCommand(), DownloadCommand()]
    )
    
    cmd.register_parser(parser)

    args = parser.parse_args()

    if hasattr(args, 'handle'):
        with UmaskWrapper():
            try:
                args.handle(args)
            except Exception as err:
                raise Exception(
                    f"[ERROR] Refer FAQ if a known issue: {AIT_FAQ_HOME}"
                ) from err


if __name__ == "__main__":
    main()
