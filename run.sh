model=yolov5s-visdrone-feax2-qat5


python test.py -i model/onnx_modify/${model}_rm_reshape.onnx --qnt --eval_perf  --test --eval --save_txt 