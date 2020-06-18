import tensorrt as trt
import os

TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network() as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            # 没有1个G的cuda 1<<33
            # 没有512MB的cuda 1<<32
            # 有256的cuda 1<<31 3.0s
            # 128 2.7
            builder.max_workspace_size = 1 << 26# 调节使用gpu-memory的大小，对执行速度影响不大
            builder.max_batch_size = 1
            builder.fp16_mode=True # 能提速

            print(builder.platform_has_fast_fp16)
            print(builder.platform_has_fast_int8)
            print(builder.int8_mode)
            print(builder.fp16_mode)

            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

            print('network layers ',network.num_layers)

            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

if __name__ == '__main__':
    # model_path = './weights/yolov3-myyolov3_99_0.96_warehouse_2.onnx'
    model_path = './weights/yolov3-mytiny_98_0.96_warehouse.onnx'
    # trt_engine = './weights/yolov3-myyolov3_99_0.96_warehouse_5.trt'
    trt_engine = './weights/yolov3-mytiny_98_0.96_warehouse_3.trt'
    # model_path = './weights/my_test.onnx'
    # trt_engine = './weights/my_test.trt'
    get_engine(model_path,trt_engine)
