¿½ 
¥
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
³
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.02unknown8
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
Ë
7ActorDistributionNetwork/EncodingNetwork/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*H
shared_name97ActorDistributionNetwork/EncodingNetwork/dense_6/kernel
Ä
KActorDistributionNetwork/EncodingNetwork/dense_6/kernel/Read/ReadVariableOpReadVariableOp7ActorDistributionNetwork/EncodingNetwork/dense_6/kernel*
_output_shapes
:	*
dtype0
Ã
5ActorDistributionNetwork/EncodingNetwork/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75ActorDistributionNetwork/EncodingNetwork/dense_6/bias
¼
IActorDistributionNetwork/EncodingNetwork/dense_6/bias/Read/ReadVariableOpReadVariableOp5ActorDistributionNetwork/EncodingNetwork/dense_6/bias*
_output_shapes	
:*
dtype0
Ì
7ActorDistributionNetwork/EncodingNetwork/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*H
shared_name97ActorDistributionNetwork/EncodingNetwork/dense_7/kernel
Å
KActorDistributionNetwork/EncodingNetwork/dense_7/kernel/Read/ReadVariableOpReadVariableOp7ActorDistributionNetwork/EncodingNetwork/dense_7/kernel* 
_output_shapes
:
*
dtype0
Ã
5ActorDistributionNetwork/EncodingNetwork/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75ActorDistributionNetwork/EncodingNetwork/dense_7/bias
¼
IActorDistributionNetwork/EncodingNetwork/dense_7/bias/Read/ReadVariableOpReadVariableOp5ActorDistributionNetwork/EncodingNetwork/dense_7/bias*
_output_shapes	
:*
dtype0
õ
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*]
shared_nameNLActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel
î
`ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel/Read/ReadVariableOpReadVariableOpLActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel*
_output_shapes
:	*
dtype0
ì
JActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias
å
^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias/Read/ReadVariableOpReadVariableOpJActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias*
_output_shapes
:*
dtype0

NoOpNoOp
¯
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ê
valueàBÝ BÖ
T

train_step
metadata
model_variables
_all_assets

signatures
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 
*
0
1
2
	3

4
5

0
 
yw
VARIABLE_VALUE7ActorDistributionNetwork/EncodingNetwork/dense_6/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE5ActorDistributionNetwork/EncodingNetwork/dense_6/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE7ActorDistributionNetwork/EncodingNetwork/dense_7/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE5ActorDistributionNetwork/EncodingNetwork/dense_7/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUELActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEJActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE

ref
1
z
_encoder
_projection_networks
trainable_variables
regularization_losses
	variables
	keras_api
n
_postprocessing_layers
trainable_variables
regularization_losses
	variables
	keras_api
i
_projection_layer
trainable_variables
regularization_losses
	variables
	keras_api
*
0
1
2
	3

4
5
 
*
0
1
2
	3

4
5
­
non_trainable_variables
layer_metrics
trainable_variables
 metrics
regularization_losses

!layers
	variables
"layer_regularization_losses

#0
$1
%2

0
1
2
	3
 

0
1
2
	3
­
&non_trainable_variables
'layer_metrics
trainable_variables
(metrics
regularization_losses

)layers
	variables
*layer_regularization_losses
h


kernel
bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api


0
1
 


0
1
­
/non_trainable_variables
0layer_metrics
trainable_variables
1metrics
regularization_losses

2layers
	variables
3layer_regularization_losses
 
 
 

0
1
 
R
4trainable_variables
5regularization_losses
6	variables
7	keras_api
h

kernel
bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
h

kernel
	bias
<trainable_variables
=regularization_losses
>	variables
?	keras_api
 
 
 

#0
$1
%2
 


0
1
 


0
1
­
@non_trainable_variables
Alayer_metrics
+trainable_variables
Bmetrics
,regularization_losses

Clayers
-	variables
Dlayer_regularization_losses
 
 
 

0
 
 
 
 
­
Enon_trainable_variables
Flayer_metrics
4trainable_variables
Gmetrics
5regularization_losses

Hlayers
6	variables
Ilayer_regularization_losses

0
1
 

0
1
­
Jnon_trainable_variables
Klayer_metrics
8trainable_variables
Lmetrics
9regularization_losses

Mlayers
:	variables
Nlayer_regularization_losses

0
	1
 

0
	1
­
Onon_trainable_variables
Player_metrics
<trainable_variables
Qmetrics
=regularization_losses

Rlayers
>	variables
Slayer_regularization_losses
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
l
action_0/discountPlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
w
action_0/observationPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
j
action_0/rewardPlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
m
action_0/step_typePlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ò
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_type7ActorDistributionNetwork/EncodingNetwork/dense_6/kernel5ActorDistributionNetwork/EncodingNetwork/dense_6/bias7ActorDistributionNetwork/EncodingNetwork/dense_7/kernel5ActorDistributionNetwork/EncodingNetwork/dense_7/biasLActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernelJActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_35347718
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
û
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_35347730
Ü
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_35347752

StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_35347745
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
´
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpKActorDistributionNetwork/EncodingNetwork/dense_6/kernel/Read/ReadVariableOpIActorDistributionNetwork/EncodingNetwork/dense_6/bias/Read/ReadVariableOpKActorDistributionNetwork/EncodingNetwork/dense_7/kernel/Read/ReadVariableOpIActorDistributionNetwork/EncodingNetwork/dense_7/bias/Read/ReadVariableOp`ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel/Read/ReadVariableOp^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias/Read/ReadVariableOpConst*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_35348129
£
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable7ActorDistributionNetwork/EncodingNetwork/dense_6/kernel5ActorDistributionNetwork/EncodingNetwork/dense_6/bias7ActorDistributionNetwork/EncodingNetwork/dense_7/kernel5ActorDistributionNetwork/EncodingNetwork/dense_7/biasLActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernelJActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_35348160ÀÊ
¿

ë
,__inference_function_with_signature_35347696
	step_type

reward
discount
observation
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *3
f.R,
*__inference_polymorphic_action_fn_353476812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_name0/step_type:MI
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
0/reward:OK
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
0/discount:VR
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_name0/observation
4

__inference_<lambda>_35347348*
_input_shapes 
¸®	
	
*__inference_polymorphic_action_fn_35347681
	time_step
time_step_1
time_step_2
time_step_3S
Oactordistributionnetwork_encodingnetwork_dense_6_matmul_readvariableop_resourceT
Pactordistributionnetwork_encodingnetwork_dense_6_biasadd_readvariableop_resourceS
Oactordistributionnetwork_encodingnetwork_dense_7_matmul_readvariableop_resourceT
Pactordistributionnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resourceh
dactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resourcei
eactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource
identity¢GActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp¢FActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOp¢GActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp¢FActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp¢\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp¢[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOpÇ
9ActorDistributionNetwork/EncodingNetwork/flatten_29/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2;
9ActorDistributionNetwork/EncodingNetwork/flatten_29/Const
;ActorDistributionNetwork/EncodingNetwork/flatten_29/ReshapeReshapetime_step_3BActorDistributionNetwork/EncodingNetwork/flatten_29/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;ActorDistributionNetwork/EncodingNetwork/flatten_29/Reshape¡
FActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_6_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOpÅ
7ActorDistributionNetwork/EncodingNetwork/dense_6/MatMulMatMulDActorDistributionNetwork/EncodingNetwork/flatten_29/Reshape:output:0NActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7ActorDistributionNetwork/EncodingNetwork/dense_6/MatMul 
GActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpÆ
8ActorDistributionNetwork/EncodingNetwork/dense_6/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_6/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8ActorDistributionNetwork/EncodingNetwork/dense_6/BiasAddì
5ActorDistributionNetwork/EncodingNetwork/dense_6/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5ActorDistributionNetwork/EncodingNetwork/dense_6/Relu¢
FActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpÄ
7ActorDistributionNetwork/EncodingNetwork/dense_7/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_6/Relu:activations:0NActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7ActorDistributionNetwork/EncodingNetwork/dense_7/MatMul 
GActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpÆ
8ActorDistributionNetwork/EncodingNetwork/dense_7/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_7/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8ActorDistributionNetwork/EncodingNetwork/dense_7/BiasAddì
5ActorDistributionNetwork/EncodingNetwork/dense_7/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5ActorDistributionNetwork/EncodingNetwork/dense_7/Reluà
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOpReadVariableOpdactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02]
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_7/Relu:activations:0cActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMulÞ
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOpReadVariableOpeactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02^
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAddBiasAddVActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul:product:0dActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2O
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAddº
:ActorDistributionNetwork/TanhNormalProjectionNetwork/ConstConst*
_output_shapes
: *
dtype0*
value	B :2<
:ActorDistributionNetwork/TanhNormalProjectionNetwork/Const×
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2F
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dimþ
:ActorDistributionNetwork/TanhNormalProjectionNetwork/splitSplitMActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dim:output:0VActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd:output:0*
T0*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2<
:ActorDistributionNetwork/TanhNormalProjectionNetwork/splitÙ
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shapeË
<ActorDistributionNetwork/TanhNormalProjectionNetwork/ReshapeReshapeCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:0KActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<ActorDistributionNetwork/TanhNormalProjectionNetwork/Reshapeò
8ActorDistributionNetwork/TanhNormalProjectionNetwork/ExpExpCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Const°
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2s
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeÕ
±ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape<ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:y:0*
T0*
_output_shapes
:2´
±ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeØ
¿ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2Â
¿ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackÓ
ÁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Ä
ÁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Ó
ÁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ä
ÁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2ñ	
¹ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSliceºActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ÈActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0ÊActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0ÊActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2¼
¹ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceø
»ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1PackÂActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2¾
»ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1·
·ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2º
·ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisì
²ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2ºActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ÄActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0ÀActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2µ
²ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2¢
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack
¡ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2¤
¡ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1
¡ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2¤
¡ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Ð
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlice»ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0¨ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0ªActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0ªActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceÏ
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShapeEActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape:output:0*
T0*
_output_shapes
:2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeÀ
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2{
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackÍ
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Ä
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Ç
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicetActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceº
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgs¢ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0|ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs÷
WActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2Y
WActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/Const¸
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zerosFillxActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0`ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2S
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zerosé
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/ones³
±ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2´
±ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeæ
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeroÃ
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast/xConst*
_output_shapes
: *
dtype0*
valueB 2        2=
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast/xô
9ActorDistributionNetwork/TanhNormalProjectionNetwork/CastCastDActorDistributionNetwork/TanhNormalProjectionNetwork/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2;
9ActorDistributionNetwork/TanhNormalProjectionNetwork/CastÇ
=ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB 2      ð?2?
=ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1/xú
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1CastFActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2=
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1µ
¶ActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2¹
¶ActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zeroÔ
ÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2È
ÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape	
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeÙ
ÄActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2Ç
ÄActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2¡
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape¿
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ShapeShapeZActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape
¢ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2¥
¢ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1Ã
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgsBroadcastArgs ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape:output:0«ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2¢
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2¤
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axisî
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concatConcatV2ªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0:output:0¤ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs:r0:0¦ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat¢
«ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2®
«ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean¦
­ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2°
­ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevø
»ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02¾
»ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal
ªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mulMulÄActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal:output:0¶ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2­
ªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mulô
¦ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normalAdd®ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mul:z:0´ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2©
¦ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normalò
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mulMulªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal:z:0YActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/ones:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mulä
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/addAddV2ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mul:z:0ZActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1ShapeActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add:z:0*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1
¥ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2¨
¥ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack
§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2ª
§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1
§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2ª
§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Ó&
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_sliceStridedSlice¢ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1:output:0®ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack:output:0°ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1:output:0°ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2¢
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2¢
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axisõ
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1ConcatV2§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape:output:0¨ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice:output:0¨ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1¼
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ReshapeReshapeActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add:z:0£ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshapeè
ÆActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2É
ÆActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/permÃ
ÁActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose	Transpose¢ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0ÏActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ä
ÁActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose÷
½ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ShapeShapeÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose:y:0*
T0*
_output_shapes
:2À
½ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shapeç
ËActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2Î
ËActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stackë
ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Ð
ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1ë
ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ð
ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2µ"
ÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_sliceStridedSliceÆActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape:output:0ÔActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack:output:0ÖActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1:output:0ÖActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2È
ÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_sliceÏ
ÃActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Æ
ÃActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis­
¾ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concatConcatV2ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape:output:0ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice:output:0ÌActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2Á
¾ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concatØ
¿ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ReshapeReshapeÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose:y:0ÇActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Â
¿ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape¢
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ShapeShapeÈActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape	
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2¢
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack	
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2¤
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1	
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2¤
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2­
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceStridedSliceActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape:output:0¨ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack:output:0ªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0ªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice÷
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisÑ
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concatConcatV2¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape:output:0¢ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice:output:0 ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat×
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ReshapeReshapeÈActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape
µActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulMul<ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:y:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¸
µActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulÑ
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/addAddV2¹ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mul:z:0EActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add¼
¾ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ShapeShapeActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add:z:0*
T0*
_output_shapes
:2Á
¾ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shapeé
ÌActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2Ï
ÌActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stackí
ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Ñ
ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1í
ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ñ
ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2»
ÆActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceStridedSliceÇActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape:output:0ÕActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack:output:0×ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0×ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2É
ÆActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceÑ
ÄActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Ç
ÄActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis²
¿ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concatConcatV2ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape:output:0ÏActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice:output:0ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2Â
¿ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat
ÀActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ReshapeReshapeActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add:z:0ÈActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ã
ÀActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshapeÿ
¹ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/TanhTanhÉActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¼
¹ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/Tanhÿ
ýActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/IdentityIdentity½ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ýActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/Identityú
øActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mulMulActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/Identity:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2û
øActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mulð
øActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/addAddV2üActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mul:z:0=ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2û
øActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y	
clip_by_value/MinimumMinimumüActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/add:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_valueÈ
IdentityIdentityclip_by_value:z:0H^ActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp]^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp\^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2
GActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp2
FActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOp2
GActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp2
FActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp2¼
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp2º
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp:N J
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	time_step:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	time_step
°
>
,__inference_function_with_signature_35347725

batch_size
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_get_initial_state_353477242
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
á%
þ
$__inference__traced_restore_35348160
file_prefix
assignvariableop_variableN
Jassignvariableop_1_actordistributionnetwork_encodingnetwork_dense_6_kernelL
Hassignvariableop_2_actordistributionnetwork_encodingnetwork_dense_6_biasN
Jassignvariableop_3_actordistributionnetwork_encodingnetwork_dense_7_kernelL
Hassignvariableop_4_actordistributionnetwork_encodingnetwork_dense_7_biasc
_assignvariableop_5_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_kernela
]assignvariableop_6_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_bias

identity_8¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6â
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*î
valueäBáB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
RestoreV2/shape_and_slicesÓ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ï
AssignVariableOp_1AssignVariableOpJassignvariableop_1_actordistributionnetwork_encodingnetwork_dense_6_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Í
AssignVariableOp_2AssignVariableOpHassignvariableop_2_actordistributionnetwork_encodingnetwork_dense_6_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ï
AssignVariableOp_3AssignVariableOpJassignvariableop_3_actordistributionnetwork_encodingnetwork_dense_7_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Í
AssignVariableOp_4AssignVariableOpHassignvariableop_4_actordistributionnetwork_encodingnetwork_dense_7_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ä
AssignVariableOp_5AssignVariableOp_assignvariableop_5_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6â
AssignVariableOp_6AssignVariableOp]assignvariableop_6_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpù

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_7ë

Identity_8IdentityIdentity_7:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6*
T0*
_output_shapes
: 2

Identity_8"!

identity_8Identity_8:output:0*1
_input_shapes 
: :::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
°
8
&__inference_signature_wrapper_35347730

batch_size
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *5
f0R.
,__inference_function_with_signature_353477252
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size

^
__inference_<lambda>_35347345
readvariableop_resource
identity	¢ReadVariableOpp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpj
IdentityIdentityReadVariableOp:value:0^ReadVariableOp*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2 
ReadVariableOpReadVariableOp
í¹
	
0__inference_polymorphic_distribution_fn_35348077
	step_type

reward
discount
observationS
Oactordistributionnetwork_encodingnetwork_dense_6_matmul_readvariableop_resourceT
Pactordistributionnetwork_encodingnetwork_dense_6_biasadd_readvariableop_resourceS
Oactordistributionnetwork_encodingnetwork_dense_7_matmul_readvariableop_resourceT
Pactordistributionnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resourceh
dactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resourcei
eactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource¢GActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp¢FActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOp¢GActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp¢FActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp¢\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp¢[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp¢StatefulPartitionedCallÇ
9ActorDistributionNetwork/EncodingNetwork/flatten_29/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2;
9ActorDistributionNetwork/EncodingNetwork/flatten_29/Const
;ActorDistributionNetwork/EncodingNetwork/flatten_29/ReshapeReshapeobservationBActorDistributionNetwork/EncodingNetwork/flatten_29/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;ActorDistributionNetwork/EncodingNetwork/flatten_29/Reshape¡
FActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_6_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOpÅ
7ActorDistributionNetwork/EncodingNetwork/dense_6/MatMulMatMulDActorDistributionNetwork/EncodingNetwork/flatten_29/Reshape:output:0NActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7ActorDistributionNetwork/EncodingNetwork/dense_6/MatMul 
GActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpÆ
8ActorDistributionNetwork/EncodingNetwork/dense_6/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_6/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8ActorDistributionNetwork/EncodingNetwork/dense_6/BiasAddì
5ActorDistributionNetwork/EncodingNetwork/dense_6/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5ActorDistributionNetwork/EncodingNetwork/dense_6/Relu¢
FActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpÄ
7ActorDistributionNetwork/EncodingNetwork/dense_7/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_6/Relu:activations:0NActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7ActorDistributionNetwork/EncodingNetwork/dense_7/MatMul 
GActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpÆ
8ActorDistributionNetwork/EncodingNetwork/dense_7/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_7/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8ActorDistributionNetwork/EncodingNetwork/dense_7/BiasAddì
5ActorDistributionNetwork/EncodingNetwork/dense_7/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5ActorDistributionNetwork/EncodingNetwork/dense_7/Reluà
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOpReadVariableOpdactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02]
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_7/Relu:activations:0cActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMulÞ
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOpReadVariableOpeactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02^
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAddBiasAddVActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul:product:0dActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2O
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAddº
:ActorDistributionNetwork/TanhNormalProjectionNetwork/ConstConst*
_output_shapes
: *
dtype0*
value	B :2<
:ActorDistributionNetwork/TanhNormalProjectionNetwork/Const×
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2F
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dimþ
:ActorDistributionNetwork/TanhNormalProjectionNetwork/splitSplitMActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dim:output:0VActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd:output:0*
T0*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2<
:ActorDistributionNetwork/TanhNormalProjectionNetwork/splitÙ
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shapeË
<ActorDistributionNetwork/TanhNormalProjectionNetwork/ReshapeReshapeCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:0KActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<ActorDistributionNetwork/TanhNormalProjectionNetwork/Reshapeò
8ActorDistributionNetwork/TanhNormalProjectionNetwork/ExpExpCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Const°
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2s
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeÕ
±ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape<ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:y:0*
T0*
_output_shapes
:2´
±ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeØ
¿ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2Â
¿ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackÓ
ÁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Ä
ÁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Ó
ÁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ä
ÁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2ñ	
¹ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSliceºActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ÈActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0ÊActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0ÊActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2¼
¹ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceø
»ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1PackÂActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2¾
»ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1·
·ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2º
·ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisì
²ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2ºActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ÄActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0ÀActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2µ
²ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2¢
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack
¡ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2¤
¡ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1
¡ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2¤
¡ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Ð
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlice»ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0¨ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0ªActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0ªActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceÏ
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShapeEActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape:output:0*
T0*
_output_shapes
:2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeÀ
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2{
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackÍ
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Ä
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Ç
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicetActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceº
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgs¢ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0|ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs÷
WActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2Y
WActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/Const¸
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zerosFillxActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0`ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2S
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zerosé
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/ones³
±ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2´
±ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeæ
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeroÃ
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast/xConst*
_output_shapes
: *
dtype0*
valueB 2        2=
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast/xô
9ActorDistributionNetwork/TanhNormalProjectionNetwork/CastCastDActorDistributionNetwork/TanhNormalProjectionNetwork/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2;
9ActorDistributionNetwork/TanhNormalProjectionNetwork/CastÇ
=ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB 2      ð?2?
=ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1/xú
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1CastFActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2=
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1µ
¶ActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2¹
¶ActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zero
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *$
fR
__inference__raise_353480762
StatefulPartitionedCall*k
_input_shapesZ
X:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2
GActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp2
FActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOp2
GActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp2
FActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp2¼
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp2º
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	step_type:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namereward:MI
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
discount:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameobservation

`
&__inference_signature_wrapper_35347745
unknown
identity	¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *5
f0R.
,__inference_function_with_signature_353477372
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
®®	
	
*__inference_polymorphic_action_fn_35347879
	step_type

reward
discount
observationS
Oactordistributionnetwork_encodingnetwork_dense_6_matmul_readvariableop_resourceT
Pactordistributionnetwork_encodingnetwork_dense_6_biasadd_readvariableop_resourceS
Oactordistributionnetwork_encodingnetwork_dense_7_matmul_readvariableop_resourceT
Pactordistributionnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resourceh
dactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resourcei
eactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource
identity¢GActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp¢FActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOp¢GActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp¢FActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp¢\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp¢[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOpÇ
9ActorDistributionNetwork/EncodingNetwork/flatten_29/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2;
9ActorDistributionNetwork/EncodingNetwork/flatten_29/Const
;ActorDistributionNetwork/EncodingNetwork/flatten_29/ReshapeReshapeobservationBActorDistributionNetwork/EncodingNetwork/flatten_29/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;ActorDistributionNetwork/EncodingNetwork/flatten_29/Reshape¡
FActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_6_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOpÅ
7ActorDistributionNetwork/EncodingNetwork/dense_6/MatMulMatMulDActorDistributionNetwork/EncodingNetwork/flatten_29/Reshape:output:0NActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7ActorDistributionNetwork/EncodingNetwork/dense_6/MatMul 
GActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpÆ
8ActorDistributionNetwork/EncodingNetwork/dense_6/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_6/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8ActorDistributionNetwork/EncodingNetwork/dense_6/BiasAddì
5ActorDistributionNetwork/EncodingNetwork/dense_6/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5ActorDistributionNetwork/EncodingNetwork/dense_6/Relu¢
FActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpÄ
7ActorDistributionNetwork/EncodingNetwork/dense_7/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_6/Relu:activations:0NActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7ActorDistributionNetwork/EncodingNetwork/dense_7/MatMul 
GActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpÆ
8ActorDistributionNetwork/EncodingNetwork/dense_7/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_7/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8ActorDistributionNetwork/EncodingNetwork/dense_7/BiasAddì
5ActorDistributionNetwork/EncodingNetwork/dense_7/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5ActorDistributionNetwork/EncodingNetwork/dense_7/Reluà
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOpReadVariableOpdactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02]
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_7/Relu:activations:0cActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMulÞ
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOpReadVariableOpeactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02^
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAddBiasAddVActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul:product:0dActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2O
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAddº
:ActorDistributionNetwork/TanhNormalProjectionNetwork/ConstConst*
_output_shapes
: *
dtype0*
value	B :2<
:ActorDistributionNetwork/TanhNormalProjectionNetwork/Const×
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2F
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dimþ
:ActorDistributionNetwork/TanhNormalProjectionNetwork/splitSplitMActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dim:output:0VActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd:output:0*
T0*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2<
:ActorDistributionNetwork/TanhNormalProjectionNetwork/splitÙ
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shapeË
<ActorDistributionNetwork/TanhNormalProjectionNetwork/ReshapeReshapeCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:0KActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<ActorDistributionNetwork/TanhNormalProjectionNetwork/Reshapeò
8ActorDistributionNetwork/TanhNormalProjectionNetwork/ExpExpCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Const°
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2s
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeÕ
±ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape<ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:y:0*
T0*
_output_shapes
:2´
±ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeØ
¿ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2Â
¿ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackÓ
ÁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Ä
ÁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Ó
ÁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ä
ÁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2ñ	
¹ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSliceºActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ÈActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0ÊActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0ÊActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2¼
¹ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceø
»ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1PackÂActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2¾
»ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1·
·ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2º
·ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisì
²ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2ºActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ÄActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0ÀActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2µ
²ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2¢
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack
¡ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2¤
¡ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1
¡ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2¤
¡ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Ð
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlice»ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0¨ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0ªActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0ªActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceÏ
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShapeEActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape:output:0*
T0*
_output_shapes
:2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeÀ
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2{
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackÍ
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Ä
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Ç
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicetActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceº
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgs¢ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0|ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs÷
WActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2Y
WActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/Const¸
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zerosFillxActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0`ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2S
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zerosé
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/ones³
±ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2´
±ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeæ
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeroÃ
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast/xConst*
_output_shapes
: *
dtype0*
valueB 2        2=
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast/xô
9ActorDistributionNetwork/TanhNormalProjectionNetwork/CastCastDActorDistributionNetwork/TanhNormalProjectionNetwork/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2;
9ActorDistributionNetwork/TanhNormalProjectionNetwork/CastÇ
=ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB 2      ð?2?
=ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1/xú
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1CastFActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2=
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1µ
¶ActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2¹
¶ActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zeroÔ
ÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2È
ÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape	
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeÙ
ÄActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2Ç
ÄActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2¡
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape¿
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ShapeShapeZActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape
¢ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2¥
¢ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1Ã
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgsBroadcastArgs ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape:output:0«ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2¢
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2¤
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axisî
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concatConcatV2ªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0:output:0¤ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs:r0:0¦ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat¢
«ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2®
«ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean¦
­ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2°
­ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevø
»ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02¾
»ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal
ªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mulMulÄActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal:output:0¶ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2­
ªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mulô
¦ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normalAdd®ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mul:z:0´ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2©
¦ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normalò
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mulMulªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal:z:0YActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/ones:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mulä
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/addAddV2ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mul:z:0ZActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1ShapeActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add:z:0*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1
¥ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2¨
¥ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack
§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2ª
§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1
§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2ª
§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Ó&
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_sliceStridedSlice¢ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1:output:0®ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack:output:0°ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1:output:0°ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2¢
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2¢
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axisõ
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1ConcatV2§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape:output:0¨ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice:output:0¨ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1¼
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ReshapeReshapeActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add:z:0£ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshapeè
ÆActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2É
ÆActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/permÃ
ÁActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose	Transpose¢ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0ÏActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ä
ÁActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose÷
½ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ShapeShapeÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose:y:0*
T0*
_output_shapes
:2À
½ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shapeç
ËActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2Î
ËActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stackë
ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Ð
ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1ë
ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ð
ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2µ"
ÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_sliceStridedSliceÆActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape:output:0ÔActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack:output:0ÖActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1:output:0ÖActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2È
ÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_sliceÏ
ÃActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Æ
ÃActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis­
¾ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concatConcatV2ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape:output:0ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice:output:0ÌActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2Á
¾ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concatØ
¿ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ReshapeReshapeÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose:y:0ÇActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Â
¿ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape¢
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ShapeShapeÈActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape	
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2¢
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack	
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2¤
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1	
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2¤
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2­
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceStridedSliceActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape:output:0¨ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack:output:0ªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0ªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice÷
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisÑ
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concatConcatV2¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape:output:0¢ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice:output:0 ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat×
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ReshapeReshapeÈActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape
µActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulMul<ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:y:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¸
µActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulÑ
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/addAddV2¹ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mul:z:0EActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add¼
¾ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ShapeShapeActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add:z:0*
T0*
_output_shapes
:2Á
¾ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shapeé
ÌActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2Ï
ÌActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stackí
ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Ñ
ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1í
ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ñ
ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2»
ÆActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceStridedSliceÇActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape:output:0ÕActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack:output:0×ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0×ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2É
ÆActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceÑ
ÄActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Ç
ÄActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis²
¿ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concatConcatV2ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape:output:0ÏActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice:output:0ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2Â
¿ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat
ÀActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ReshapeReshapeActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add:z:0ÈActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ã
ÀActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshapeÿ
¹ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/TanhTanhÉActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¼
¹ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/Tanhÿ
ýActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/IdentityIdentity½ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ýActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/Identityú
øActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mulMulActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/Identity:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2û
øActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mulð
øActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/addAddV2üActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mul:z:0=ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2û
øActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y	
clip_by_value/MinimumMinimumüActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/add:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_valueÈ
IdentityIdentityclip_by_value:z:0H^ActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp]^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp\^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2
GActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp2
FActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOp2
GActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp2
FActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp2¼
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp2º
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp:N J
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	step_type:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namereward:MI
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
discount:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameobservation
À
.
,__inference_function_with_signature_35347748ú
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference_<lambda>_353473482
PartitionedCall*
_input_shapes 
ý 
Ë
!__inference__traced_save_35348129
file_prefix'
#savev2_variable_read_readvariableop	V
Rsavev2_actordistributionnetwork_encodingnetwork_dense_6_kernel_read_readvariableopT
Psavev2_actordistributionnetwork_encodingnetwork_dense_6_bias_read_readvariableopV
Rsavev2_actordistributionnetwork_encodingnetwork_dense_7_kernel_read_readvariableopT
Psavev2_actordistributionnetwork_encodingnetwork_dense_7_bias_read_readvariableopk
gsavev2_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_kernel_read_readvariableopi
esavev2_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÜ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*î
valueäBáB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopRsavev2_actordistributionnetwork_encodingnetwork_dense_6_kernel_read_readvariableopPsavev2_actordistributionnetwork_encodingnetwork_dense_6_bias_read_readvariableopRsavev2_actordistributionnetwork_encodingnetwork_dense_7_kernel_read_readvariableopPsavev2_actordistributionnetwork_encodingnetwork_dense_7_bias_read_readvariableopgsavev2_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_kernel_read_readvariableopesavev2_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*O
_input_shapes>
<: : :	::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: 
ü
f
,__inference_function_with_signature_35347737
unknown
identity	¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference_<lambda>_353473452
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
É
(
&__inference_signature_wrapper_35347752
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *5
f0R.
,__inference_function_with_signature_353477482
PartitionedCall*
_input_shapes 
ñ
0
__inference__raise_35348076¢Assert/Assertù
Assert/ConstConst*
_output_shapes
: *
dtype0*¬
value¢B BUnable to make a CompositeTensor for "<tensorflow.python.ops.linalg.linear_operator_diag.LinearOperatorDiag object at 0x7ff8afc53310>" of type `<class 'tensorflow.python.ops.linalg.linear_operator_diag.LinearOperatorDiag'>`. Email `tfprobability@tensorflow.org` or file an issue on github if you would benefit from this working. (Unable to serialize: No encoder for object [Tensor("ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:0", shape=(None, 2), dtype=float32)] of type [<class 'tensorflow.python.framework.ops.Tensor'>].)2
Assert/Constt
Assert/Assert/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
Assert/Assert/condition
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*¬
value¢B BUnable to make a CompositeTensor for "<tensorflow.python.ops.linalg.linear_operator_diag.LinearOperatorDiag object at 0x7ff8afc53310>" of type `<class 'tensorflow.python.ops.linalg.linear_operator_diag.LinearOperatorDiag'>`. Email `tfprobability@tensorflow.org` or file an issue on github if you would benefit from this working. (Unable to serialize: No encoder for object [Tensor("ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:0", shape=(None, 2), dtype=float32)] of type [<class 'tensorflow.python.framework.ops.Tensor'>].)2
Assert/Assert/data_0
Assert/AssertAssert Assert/Assert/condition:output:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assert*
_input_shapes 2
Assert/AssertAssert/Assert

8
&__inference_get_initial_state_35348080

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size

8
&__inference_get_initial_state_35347724

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
¯	
­	
*__inference_polymorphic_action_fn_35348006
time_step_step_type
time_step_reward
time_step_discount
time_step_observationS
Oactordistributionnetwork_encodingnetwork_dense_6_matmul_readvariableop_resourceT
Pactordistributionnetwork_encodingnetwork_dense_6_biasadd_readvariableop_resourceS
Oactordistributionnetwork_encodingnetwork_dense_7_matmul_readvariableop_resourceT
Pactordistributionnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resourceh
dactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resourcei
eactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource
identity¢GActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp¢FActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOp¢GActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp¢FActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp¢\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp¢[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOpÇ
9ActorDistributionNetwork/EncodingNetwork/flatten_29/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2;
9ActorDistributionNetwork/EncodingNetwork/flatten_29/Const
;ActorDistributionNetwork/EncodingNetwork/flatten_29/ReshapeReshapetime_step_observationBActorDistributionNetwork/EncodingNetwork/flatten_29/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;ActorDistributionNetwork/EncodingNetwork/flatten_29/Reshape¡
FActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_6_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOpÅ
7ActorDistributionNetwork/EncodingNetwork/dense_6/MatMulMatMulDActorDistributionNetwork/EncodingNetwork/flatten_29/Reshape:output:0NActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7ActorDistributionNetwork/EncodingNetwork/dense_6/MatMul 
GActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpÆ
8ActorDistributionNetwork/EncodingNetwork/dense_6/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_6/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8ActorDistributionNetwork/EncodingNetwork/dense_6/BiasAddì
5ActorDistributionNetwork/EncodingNetwork/dense_6/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5ActorDistributionNetwork/EncodingNetwork/dense_6/Relu¢
FActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpÄ
7ActorDistributionNetwork/EncodingNetwork/dense_7/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_6/Relu:activations:0NActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7ActorDistributionNetwork/EncodingNetwork/dense_7/MatMul 
GActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpÆ
8ActorDistributionNetwork/EncodingNetwork/dense_7/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_7/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8ActorDistributionNetwork/EncodingNetwork/dense_7/BiasAddì
5ActorDistributionNetwork/EncodingNetwork/dense_7/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5ActorDistributionNetwork/EncodingNetwork/dense_7/Reluà
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOpReadVariableOpdactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02]
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_7/Relu:activations:0cActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMulÞ
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOpReadVariableOpeactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02^
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAddBiasAddVActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul:product:0dActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2O
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAddº
:ActorDistributionNetwork/TanhNormalProjectionNetwork/ConstConst*
_output_shapes
: *
dtype0*
value	B :2<
:ActorDistributionNetwork/TanhNormalProjectionNetwork/Const×
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2F
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dimþ
:ActorDistributionNetwork/TanhNormalProjectionNetwork/splitSplitMActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dim:output:0VActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd:output:0*
T0*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2<
:ActorDistributionNetwork/TanhNormalProjectionNetwork/splitÙ
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shapeË
<ActorDistributionNetwork/TanhNormalProjectionNetwork/ReshapeReshapeCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:0KActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<ActorDistributionNetwork/TanhNormalProjectionNetwork/Reshapeò
8ActorDistributionNetwork/TanhNormalProjectionNetwork/ExpExpCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Const°
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2s
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeÕ
±ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape<ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:y:0*
T0*
_output_shapes
:2´
±ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeØ
¿ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2Â
¿ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackÓ
ÁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Ä
ÁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Ó
ÁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ä
ÁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2ñ	
¹ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSliceºActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ÈActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0ÊActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0ÊActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2¼
¹ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceø
»ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1PackÂActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2¾
»ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1·
·ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2º
·ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisì
²ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2ºActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ÄActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0ÀActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2µ
²ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2¢
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack
¡ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2¤
¡ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1
¡ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2¤
¡ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Ð
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlice»ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0¨ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0ªActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0ªActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceÏ
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShapeEActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape:output:0*
T0*
_output_shapes
:2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeÀ
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2{
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackÍ
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Ä
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Ç
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicetActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceº
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgs¢ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0|ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs÷
WActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2Y
WActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/Const¸
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zerosFillxActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0`ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2S
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zerosé
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/ones³
±ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2´
±ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeæ
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeroÃ
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast/xConst*
_output_shapes
: *
dtype0*
valueB 2        2=
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast/xô
9ActorDistributionNetwork/TanhNormalProjectionNetwork/CastCastDActorDistributionNetwork/TanhNormalProjectionNetwork/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2;
9ActorDistributionNetwork/TanhNormalProjectionNetwork/CastÇ
=ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB 2      ð?2?
=ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1/xú
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1CastFActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2=
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1µ
¶ActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2¹
¶ActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zeroÔ
ÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2È
ÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape	
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeÙ
ÄActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2Ç
ÄActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2¡
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape¿
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ShapeShapeZActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape
¢ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2¥
¢ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1Ã
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgsBroadcastArgs ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape:output:0«ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2¢
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2¤
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axisî
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concatConcatV2ªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0:output:0¤ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs:r0:0¦ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat¢
«ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2®
«ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean¦
­ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2°
­ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevø
»ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02¾
»ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal
ªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mulMulÄActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal:output:0¶ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2­
ªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mulô
¦ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normalAdd®ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mul:z:0´ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2©
¦ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normalò
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mulMulªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal:z:0YActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/ones:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mulä
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/addAddV2ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mul:z:0ZActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1ShapeActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add:z:0*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1
¥ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2¨
¥ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack
§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2ª
§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1
§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2ª
§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Ó&
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_sliceStridedSlice¢ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1:output:0®ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack:output:0°ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1:output:0°ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2¢
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2¢
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axisõ
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1ConcatV2§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape:output:0¨ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice:output:0¨ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1¼
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ReshapeReshapeActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add:z:0£ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshapeè
ÆActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2É
ÆActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/permÃ
ÁActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose	Transpose¢ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0ÏActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ä
ÁActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose÷
½ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ShapeShapeÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose:y:0*
T0*
_output_shapes
:2À
½ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shapeç
ËActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2Î
ËActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stackë
ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Ð
ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1ë
ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ð
ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2µ"
ÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_sliceStridedSliceÆActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape:output:0ÔActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack:output:0ÖActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1:output:0ÖActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2È
ÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_sliceÏ
ÃActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Æ
ÃActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis­
¾ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concatConcatV2ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape:output:0ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice:output:0ÌActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2Á
¾ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concatØ
¿ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ReshapeReshapeÅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose:y:0ÇActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Â
¿ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape¢
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ShapeShapeÈActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape	
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2¢
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack	
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2¤
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1	
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2¤
¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2­
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceStridedSliceActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape:output:0¨ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack:output:0ªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0ªActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice÷
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisÑ
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concatConcatV2¡ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape:output:0¢ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice:output:0 ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat×
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ReshapeReshapeÈActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape
µActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulMul<ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:y:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¸
µActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulÑ
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/addAddV2¹ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mul:z:0EActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add¼
¾ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ShapeShapeActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add:z:0*
T0*
_output_shapes
:2Á
¾ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shapeé
ÌActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2Ï
ÌActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stackí
ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Ñ
ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1í
ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ñ
ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2»
ÆActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceStridedSliceÇActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape:output:0ÕActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack:output:0×ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0×ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2É
ÆActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceÑ
ÄActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Ç
ÄActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis²
¿ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concatConcatV2ÎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape:output:0ÏActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice:output:0ÍActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2Â
¿ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat
ÀActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ReshapeReshapeActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add:z:0ÈActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ã
ÀActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshapeÿ
¹ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/TanhTanhÉActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¼
¹ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/Tanhÿ
ýActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/IdentityIdentity½ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ýActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/Identityú
øActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mulMulActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/Identity:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2û
øActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mulð
øActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/addAddV2üActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mul:z:0=ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2û
øActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y	
clip_by_value/MinimumMinimumüActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/add:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ¿2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
clip_by_valueÈ
IdentityIdentityclip_by_value:z:0H^ActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp]^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp\^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2
GActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_6/BiasAdd/ReadVariableOp2
FActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_6/MatMul/ReadVariableOp2
GActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp2
FActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp2¼
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp2º
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp:X T
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,
_user_specified_nametime_step/discount:^Z
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_nametime_step/observation
»

å
&__inference_signature_wrapper_35347718
discount
observation

reward
	step_type
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *5
f0R.
,__inference_function_with_signature_353476962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
0/discount:VR
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_name0/observation:MI
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
0/reward:PL
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_name0/step_type"±L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ã
action¸
4

0/discount&
action_0/discount:0ÿÿÿÿÿÿÿÿÿ
>
0/observation-
action_0/observation:0ÿÿÿÿÿÿÿÿÿ
0
0/reward$
action_0/reward:0ÿÿÿÿÿÿÿÿÿ
6
0/step_type'
action_0/step_type:0ÿÿÿÿÿÿÿÿÿ:
action0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:¨
Í

train_step
metadata
model_variables
_all_assets

signatures

Taction
Udistribution
Vget_initial_state
Wget_metadata
Xget_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
K
0
1
2
	3

4
5"
trackable_tuple_wrapper
'
0"
trackable_list_wrapper
`

Yaction
Zget_initial_state
[get_train_step
\get_metadata"
signature_map
J:H	27ActorDistributionNetwork/EncodingNetwork/dense_6/kernel
D:B25ActorDistributionNetwork/EncodingNetwork/dense_6/bias
K:I
27ActorDistributionNetwork/EncodingNetwork/dense_7/kernel
D:B25ActorDistributionNetwork/EncodingNetwork/dense_7/bias
_:]	2LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel
X:V2JActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias
1
ref
1"
trackable_tuple_wrapper
é
_encoder
_projection_networks
trainable_variables
regularization_losses
	variables
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"²
_tf_keras_layer{"class_name": "ActorDistributionNetwork", "name": "ActorDistributionNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Ë
_postprocessing_layers
trainable_variables
regularization_losses
	variables
	keras_api
___call__
*`&call_and_return_all_conditional_losses" 
_tf_keras_layer{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Þ
_projection_layer
trainable_variables
regularization_losses
	variables
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"¸
_tf_keras_layer{"class_name": "TanhNormalProjectionNetwork", "name": "TanhNormalProjectionNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
J
0
1
2
	3

4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
	3

4
5"
trackable_list_wrapper
­
non_trainable_variables
layer_metrics
trainable_variables
 metrics
regularization_losses

!layers
	variables
"layer_regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
5
#0
$1
%2"
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
­
&non_trainable_variables
'layer_metrics
trainable_variables
(metrics
regularization_losses

)layers
	variables
*layer_regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object



kernel
bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api
c__call__
*d&call_and_return_all_conditional_losses"Ý
_tf_keras_layerÃ{"class_name": "Dense", "name": "projection_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "projection_layer", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}}
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
­
/non_trainable_variables
0layer_metrics
trainable_variables
1metrics
regularization_losses

2layers
	variables
3layer_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
è
4trainable_variables
5regularization_losses
6	variables
7	keras_api
e__call__
*f&call_and_return_all_conditional_losses"Ù
_tf_keras_layer¿{"class_name": "Flatten", "name": "flatten_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_29", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


kernel
bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
g__call__
*h&call_and_return_all_conditional_losses"Û
_tf_keras_layerÁ{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4]}}


kernel
	bias
<trainable_variables
=regularization_losses
>	variables
?	keras_api
i__call__
*j&call_and_return_all_conditional_losses"ß
_tf_keras_layerÅ{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
#0
$1
%2"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
­
@non_trainable_variables
Alayer_metrics
+trainable_variables
Bmetrics
,regularization_losses

Clayers
-	variables
Dlayer_regularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Enon_trainable_variables
Flayer_metrics
4trainable_variables
Gmetrics
5regularization_losses

Hlayers
6	variables
Ilayer_regularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Jnon_trainable_variables
Klayer_metrics
8trainable_variables
Lmetrics
9regularization_losses

Mlayers
:	variables
Nlayer_regularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
­
Onon_trainable_variables
Player_metrics
<trainable_variables
Qmetrics
=regularization_losses

Rlayers
>	variables
Slayer_regularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
2
*__inference_polymorphic_action_fn_35347879
*__inference_polymorphic_action_fn_35348006±
ª²¦
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults¢
¢ 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
0__inference_polymorphic_distribution_fn_35348077±
ª²¦
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults¢
¢ 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
&__inference_get_initial_state_35348080¦
²
FullArgSpec!
args
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³B°
__inference_<lambda>_35347348"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³B°
__inference_<lambda>_35347345"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ôBñ
&__inference_signature_wrapper_35347718
0/discount0/observation0/reward0/step_type"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÐBÍ
&__inference_signature_wrapper_35347730
batch_size"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÂB¿
&__inference_signature_wrapper_35347745"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÂB¿
&__inference_signature_wrapper_35347752"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2çä
Û²×
FullArgSpecU
argsMJ
jself
jobservations
j	step_type
jnetwork_state

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2çä
Û²×
FullArgSpecU
argsMJ
jself
jobservations
j	step_type
jnetwork_state

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ãà
×²Ó
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
¢ 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ãà
×²Ó
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
¢ 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2ÑÎ
Å²Á
FullArgSpec?
args74
jself
jinputs
j
outer_rank

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2ÑÎ
Å²Á
FullArgSpec?
args74
jself
jinputs
j
outer_rank

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 <
__inference_<lambda>_35347345¢

¢ 
ª " 	5
__inference_<lambda>_35347348¢

¢ 
ª "ª S
&__inference_get_initial_state_35348080)"¢
¢


batch_size 
ª "¢ ð
*__inference_polymorphic_action_fn_35347879Á	
Þ¢Ú
Ò¢Î
Æ²Â
TimeStep,
	step_type
	step_typeÿÿÿÿÿÿÿÿÿ&
reward
rewardÿÿÿÿÿÿÿÿÿ*
discount
discountÿÿÿÿÿÿÿÿÿ4
observation%"
observationÿÿÿÿÿÿÿÿÿ
¢ 
ª "V²S

PolicyStep*
action 
actionÿÿÿÿÿÿÿÿÿ
state¢ 
info¢ 
*__inference_polymorphic_action_fn_35348006é	
¢
ú¢ö
î²ê
TimeStep6
	step_type)&
time_step/step_typeÿÿÿÿÿÿÿÿÿ0
reward&#
time_step/rewardÿÿÿÿÿÿÿÿÿ4
discount(%
time_step/discountÿÿÿÿÿÿÿÿÿ>
observation/,
time_step/observationÿÿÿÿÿÿÿÿÿ
¢ 
ª "V²S

PolicyStep*
action 
actionÿÿÿÿÿÿÿÿÿ
state¢ 
info¢ ¢
0__inference_polymorphic_distribution_fn_35348077í	
Þ¢Ú
Ò¢Î
Æ²Â
TimeStep,
	step_type
	step_typeÿÿÿÿÿÿÿÿÿ&
reward
rewardÿÿÿÿÿÿÿÿÿ*
discount
discountÿÿÿÿÿÿÿÿÿ4
observation%"
observationÿÿÿÿÿÿÿÿÿ
¢ 
ª "
 ¿
&__inference_signature_wrapper_35347718	
Ø¢Ô
¢ 
ÌªÈ
.

0/discount 

0/discountÿÿÿÿÿÿÿÿÿ
8
0/observation'$
0/observationÿÿÿÿÿÿÿÿÿ
*
0/reward
0/rewardÿÿÿÿÿÿÿÿÿ
0
0/step_type!
0/step_typeÿÿÿÿÿÿÿÿÿ"/ª,
*
action 
actionÿÿÿÿÿÿÿÿÿa
&__inference_signature_wrapper_3534773070¢-
¢ 
&ª#
!

batch_size

batch_size "ª Z
&__inference_signature_wrapper_353477450¢

¢ 
ª "ª

int64
int64 	>
&__inference_signature_wrapper_35347752¢

¢ 
ª "ª 