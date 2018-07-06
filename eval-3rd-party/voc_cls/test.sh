export PYTHONPATH=:/data/chongruo-data/deepblur/philkr_eval/caffe/python
CUDA_VISIBLE_DEVICES=6 python3 src/train_cls.py \
    ./models/caffenet_deploy.prototxt \
    ./scenenet_5m_ep26_rs.caffemodel \
    -nit 80000 \
    -lr 0.001 \
    -step 10000 \
    --scale 0.003921 \

    # # !!! 67.1 : 8w, 0.001 step:1w
    # #--random-from pool
    # #--train-from pool5
#     #--output-dir ./random_conv12/
