#!/usr/bin/env bash
# 统一编排 URetinex-Net++ 的两个 decomposition、unfolding 与 adjustment 阶段。

set -euo pipefail

# 无论从哪个目录调用，都以项目根目录解析 Python 入口和输出目录。
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

usage() {
    cat <<'EOF'
用法：
  bash run_reproduction.sh --dataset <lol-v1|lol-v2-syn|lol-v2-real> \
      --data-root <数据根目录> --experiment <实验名> [选项]

选项：
  --gpu <物理 GPU 编号>          单卡顺序执行四个阶段；未给出时沿用外部 CUDA_VISIBLE_DEVICES。
  --parallel-decom               并行训练两个 decomposition 阶段；可使用同一张或不同 GPU。
  --decom-low-gpu <物理 GPU 编号>
  --decom-high-gpu <物理 GPU 编号>
  --num-workers <数量>           每个 DataLoader 的 worker 数，默认 0。
  --python <命令>                Python 解释器，默认使用 $PYTHON_BIN 或 python。
  -h, --help                     显示本帮助。

说明：
  单卡并行时可只传 --parallel-decom --gpu 0，两个进程会共享该卡。
  这与打开两个终端分别训练等价；请根据显存余量决定是否启用。
EOF
}

dataset=""
data_root=""
experiment=""
gpu=""
decom_low_gpu=""
decom_high_gpu=""
parallel_decom=false
num_workers=0
python_bin="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            dataset="$2"
            shift 2
            ;;
        --data-root)
            data_root="$2"
            shift 2
            ;;
        --experiment)
            experiment="$2"
            shift 2
            ;;
        --gpu)
            gpu="$2"
            shift 2
            ;;
        --decom-low-gpu)
            decom_low_gpu="$2"
            shift 2
            ;;
        --decom-high-gpu)
            decom_high_gpu="$2"
            shift 2
            ;;
        --parallel-decom)
            parallel_decom=true
            shift
            ;;
        --num-workers)
            num_workers="$2"
            shift 2
            ;;
        --python)
            python_bin="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数：$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$dataset" || -z "$data_root" || -z "$experiment" ]]; then
    echo "--dataset、--data-root 和 --experiment 均为必填参数。" >&2
    usage >&2
    exit 2
fi

case "$dataset" in
    lol-v1|lol-v2-syn|lol-v2-real) ;;
    *)
        echo "不支持的数据集：$dataset" >&2
        exit 2
        ;;
esac

common_args=(
    --dataset "$dataset"
    --data-root "$data_root"
    --experiment "$experiment"
    --num-workers "$num_workers"
)

run_stage() {
    local stage="$1"
    local stage_gpu="$2"
    echo "[run_reproduction] stage=$stage dataset=$dataset experiment=$experiment"
    if [[ -n "$stage_gpu" ]]; then
        CUDA_VISIBLE_DEVICES="$stage_gpu" "$python_bin" run_reproduction.py \
            --stage "$stage" "${common_args[@]}"
    else
        "$python_bin" run_reproduction.py --stage "$stage" "${common_args[@]}"
    fi
}

if [[ "$parallel_decom" == true ]]; then
    # 未单独指定时复用 --gpu；未给 --gpu 时遵从两个进程继承的外部可见卡设置。
    decom_low_gpu="${decom_low_gpu:-$gpu}"
    decom_high_gpu="${decom_high_gpu:-$gpu}"
    if [[ -n "$decom_low_gpu" && "$decom_low_gpu" == "$decom_high_gpu" ]]; then
        echo "[run_reproduction] 单卡并行：decom_low 和 decom_high 共享物理 GPU $decom_low_gpu。"
    fi

    run_stage decom_low "$decom_low_gpu" &
    decom_low_pid=$!
    run_stage decom_high "$decom_high_gpu" &
    decom_high_pid=$!

    low_status=0
    high_status=0
    wait "$decom_low_pid" || low_status=$?
    wait "$decom_high_pid" || high_status=$?
    if [[ "$low_status" -ne 0 || "$high_status" -ne 0 ]]; then
        echo "至少一个 decomposition 阶段失败；不会继续执行 unfolding。" >&2
        exit 1
    fi
else
    run_stage decom_low "$gpu"
    run_stage decom_high "$gpu"
fi

run_stage unfold "$gpu"
run_stage adjust "$gpu"

echo "[run_reproduction] 四个阶段已完成。最终测试请显式运行 test_reproduction.py 并传入 models/best_G.pth。"
