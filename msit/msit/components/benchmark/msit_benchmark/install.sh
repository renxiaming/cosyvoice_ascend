#!/bin/bash
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

declare -i ret_ok=0
declare -i ret_run_failed=1

WHL_BASE_URL=$(python3 -c "from components.utils.install import get_public_url; print(get_public_url('whl_base_url'))")
TOOLS_BAS_URL_SUFFIX=$(python3 -c "from components.utils.install import get_public_url; print(get_public_url('tools_base_url'))")
TOOLS_BAS_URL="git+$TOOLS_BAS_URL_SUFFIX"

download_and_install_aclruntime() {
    ACLRUNTIME_VERSION=`pip3 show aclruntime | awk '/Version: /{print $2}'`

    if [ "$arg_force_reinstall" = "--force-reinstall" ]; then
        echo "Force reinstall aclruntime"
    elif [ "$ACLRUNTIME_VERSION" = "0.0.2" ]; then
        echo "aclruntime==0.0.2 already installed, skip"
        return
    fi

    echo "download and install aclruntime"
    PYTHON3_MINI_VERSION=`python3 --version | cut -d'.' -f 2`
    PYTHON3_MINI_VERSION=`python3 --version | cut -d'.' -f 2`
    if [ "$PYTHON3_MINI_VERSION" = "7" ]; then
        SUB_SUFFIX="m"
    else
        SUB_SUFFIX=""
    fi
    echo "PYTHON3_MINI_VERSION=$PYTHON3_MINI_VERSION, SUB_SUFFIX=$SUB_SUFFIX"

    WHL_NAME="aclruntime-0.0.2-cp3${PYTHON3_MINI_VERSION}-cp3${PYTHON3_MINI_VERSION}${SUB_SUFFIX}-linux_$(uname -m).whl"
    echo "WHL_NAME=$WHL_NAME, URL=${WHL_BASE_URL}${WHL_NAME}"
    wget --no-check-certificate -c "${WHL_BASE_URL}${WHL_NAME}" && pip3 install $WHL_NAME $arg_force_reinstall && rm -f $WHL_NAME
    if [ $? -ne 0 ]; then
        echo "Downloading or installing from whl failed, will install from source code"
        pip3 install -v "${TOOLS_BAS_URL}#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend" --force-reinstall
    fi
}

download_and_install_ais_bench() {
    AIS_BENCH_VERSION=`pip3 show ais_bench | awk '/Version: /{print $2}'`

    if [ "$arg_force_reinstall" = "--force-reinstall" ]; then
        echo "Force reinstall ais_bench"
    elif [ "$AIS_BENCH_VERSION" = "0.0.2" ]; then
        echo "ais_bench==0.0.2 already installed, skip"
        return
    fi

    WHL_NAME="ais_bench-0.0.2-py3-none-any.whl"
    echo "WHL_NAME=$WHL_NAME, URL=${WHL_BASE_URL}${WHL_NAME}"
    wget --no-check-certificate -c "${WHL_BASE_URL}${WHL_NAME}" && pip3 install $WHL_NAME $arg_force_reinstall && rm -f $WHL_NAME
    if [ $? -ne 0 ]; then
        echo "Downloading or installing from whl failed, will install from source code"
        pip3 install -v "${TOOLS_BAS_URL}#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend" --force-reinstall
    fi
}

ret=0
download_and_install_aclruntime
ret=$(( $ret + $? ))

download_and_install_ais_bench
ret=$(( $ret + $? ))

exit $ret