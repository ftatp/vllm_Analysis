sudo /usr/local/cuda-12/bin/nsys profile -o profile_nsys \
--gpu-metrics-device=0 --cpuctxsw=none --force-overwrite true \
--trace=cuda,nvtx \
--cuda-graph-trace=node \
~/anaconda3/envs/vllm/bin/python $1