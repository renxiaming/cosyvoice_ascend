#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
if [ ! -d "$ASCEND_HOME_PATH/" ]; then
    echo "ASCEND_HOME_PATH not exist.\
    Please check that the cann package is installed.\
    Please run 'source set_env.sh' in the CANN installation path."
    exit 1
fi

src_dir="$ASCEND_HOME_PATH/python/site-packages/msmodelslim/"
script_dir=$(cd $(dirname $0);pwd)$"/msmodelslim"
echo "collect packages from CANN installation path: $src_dir, copy to $script_dir"

for file in $(find $src_dir -type f -name "*.so"); do
    file_name=$(basename $file)
    base_file=${file##$src_dir}
    base_file=${base_file%%$file_name}

    dst_dir=$script_dir/$base_file
    if [ ! -d "$dst_dir" ]; then
        mkdir $dst_dir
        touch $dst_dir"__init__.py"
        chmod 640 $dst_dir"__init__.py"
    fi
    cp $file $dst_dir
    chmod 550 $dst_dir$file_name
done

pip install .