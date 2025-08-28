import numpy
import onnx
from onnx import numpy_helper, TensorProto
from onnx import helper

class ChangeOnnx:
    def __init__(self,onnx_path,save_path,FirstConvName,FinalConcatName):
        self.onnx_path=onnx_path
        self.save_path=save_path
        self.FirstConvName=FirstConvName
        self.FinalConcatName=FinalConcatName
        self.batch_size=batch_size
        self.model=onnx.load(onnx_path)

    def change_onnx(self):
        self.change_dim()
        self.add_slice()
        onnx.save(self.model,self.save_path)

    #维度顺序
    def change_dim(self):

        # 重命名输入输出节点
        self.model.graph.input[0].name = 'input'
        self.model.graph.output[0].name = 'output'

        # 保存原维度
        OriginalInputDim = self.model.graph.input[0].type.tensor_type.shape
        OriginalOutputDim = self.model.graph.output[0].type.tensor_type.shape

        FirstConv = -1
        FinalConcat = -1

        # 查找对应卷积和全连接层
        for i, node in enumerate(self.model.graph.node, start=0):
            if node.name == self.FirstConvName:
                FirstConv = i
                print(f"FirstConv in {i}")
            if node.name == self.FinalConcatName:
                FinalConcat = i
                print(f"FinalConact in {i}")

        if FirstConv == -1:
            raise ValueError(f"未找到{self.FirstConvName}节点，FirstConv未赋值")
        if FinalConcat == -1:
            raise ValueError(f"未找到{self.FinalConcatName}节点，FinalConcat未赋值")

        # 修改对应输入输出名
        self.model.graph.node[FirstConv].input[0] = 'input'
        self.model.graph.node[FinalConcat].output[0] = 'output'

        # 形状推理/更新onnx
        self.model = onnx.shape_inference.infer_shapes(self.model)
        onnx.checker.check_model(self.model)

        # 添加转置层
        transpose_input_node = helper.make_node(
            'Transpose',
            inputs=['input'],
            outputs=['input0'],
            perm=[0, 3, 1, 2],
            name='Transposein'
        )
        transpose_output_node = helper.make_node(
            'Transpose',
            inputs=['output0'],
            outputs=['output'],
            perm=[0, 2, 1],
            name='Transposeout'
        )

        self.model.graph.node[FirstConv].input[0] = 'input0'
        self.model.graph.node[FinalConcat].output[0] = 'output0'

        # 插入转置节点
        self.model.graph.node.insert(0, transpose_input_node)
        self.model.graph.node.append(transpose_output_node)

        # 更新输入\输出形状信息
        NewInputDim = helper.make_tensor_value_info(
            'input',
            TensorProto.FLOAT,
            [
                OriginalInputDim.dim[0].dim_value,
                OriginalInputDim.dim[2].dim_value,
                OriginalInputDim.dim[3].dim_value,
                OriginalInputDim.dim[1].dim_value
            ]
        )
        NewOutputDim = helper.make_tensor_value_info(
            'output',
            TensorProto.FLOAT,
            [
                OriginalOutputDim.dim[0].dim_value,
                OriginalOutputDim.dim[2].dim_value,
                OriginalOutputDim.dim[1].dim_value
            ]
        )
        self.model.graph.input[0].CopyFrom(NewInputDim)
        self.model.graph.output[0].CopyFrom(NewOutputDim)

        self.model = onnx.shape_inference.infer_shapes(self.model)
        onnx.checker.check_model(self.model)

        return self



    def add_slice(self):

        TransposeOut=-1

        for i,node in enumerate(self.model.graph.node,start=0):
            if node.name=='Transposeout':
                TransposeOut=i
                print(f"TransposeOut in {i}")

        self.model.graph.node[TransposeOut].output[0]='slice_input'

        starts=numpy_helper.from_array(numpy.array([4],dtype=numpy.int64),name='starts')
        ends=numpy_helper.from_array(numpy.array([26],dtype=numpy.int64),name='ends')
        axes=numpy_helper.from_array(numpy.array([2],dtype=numpy.int64),name='axes')
        steps=numpy_helper.from_array(numpy.array([1],dtype=numpy.int64),name='steps')

        SliceNode=helper.make_node(
            'Slice',
            inputs=['slice_input','starts','ends','axes','steps'],
            outputs=['output'],
            name='Slice'
        )

        self.model.graph.initializer.extend([starts,ends,axes,steps])

        self.model.graph.node.append(SliceNode)

        self.model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 22

        self.model=onnx.shape_inference.infer_shapes(self.model)
        onnx.checker.check_model(self.model)

        return self
if __name__=="__main__":
    onnx_path=r"C:\Users\28921\Desktop\ultralytics-main\sentry\train1\runs\pose\train9\weights\best.onnx"
    save_path=r"C:\Users\28921\Desktop\ultralytics-main\tool\test\test.onnx"

    FirstConvName="/model.0/conv/Conv"
    FinalConcatName="/model.22/Concat_6"

    batch_size=4

    changeonnx=ChangeOnnx(onnx_path,save_path,FirstConvName,FinalConcatName)
    changeonnx.change_onnx()