�9
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
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
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
<
Selu
features"T
activations"T"
Ttype:
2
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02v2.10.0-0-g359c3cdfc5f8��4
�
,transformer_block/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*=
shared_name.,transformer_block/layer_normalization_1/beta
�
@transformer_block/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp,transformer_block/layer_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
-transformer_block/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*>
shared_name/-transformer_block/layer_normalization_1/gamma
�
Atransformer_block/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp-transformer_block/layer_normalization_1/gamma*
_output_shapes	
:�*
dtype0
�
*transformer_block/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*transformer_block/layer_normalization/beta
�
>transformer_block/layer_normalization/beta/Read/ReadVariableOpReadVariableOp*transformer_block/layer_normalization/beta*
_output_shapes	
:�*
dtype0
�
+transformer_block/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*<
shared_name-+transformer_block/layer_normalization/gamma
�
?transformer_block/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp+transformer_block/layer_normalization/gamma*
_output_shapes	
:�*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
��*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
��*
dtype0
�
<transformer_block/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*M
shared_name><transformer_block/multi_head_attention/attention_output/bias
�
Ptransformer_block/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp<transformer_block/multi_head_attention/attention_output/bias*
_output_shapes	
:�*
dtype0
�
>transformer_block/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*O
shared_name@>transformer_block/multi_head_attention/attention_output/kernel
�
Rtransformer_block/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp>transformer_block/multi_head_attention/attention_output/kernel*$
_output_shapes
:��*
dtype0
�
1transformer_block/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*B
shared_name31transformer_block/multi_head_attention/value/bias
�
Etransformer_block/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp1transformer_block/multi_head_attention/value/bias*
_output_shapes
:	�*
dtype0
�
3transformer_block/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*D
shared_name53transformer_block/multi_head_attention/value/kernel
�
Gtransformer_block/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp3transformer_block/multi_head_attention/value/kernel*$
_output_shapes
:��*
dtype0
�
/transformer_block/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*@
shared_name1/transformer_block/multi_head_attention/key/bias
�
Ctransformer_block/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp/transformer_block/multi_head_attention/key/bias*
_output_shapes
:	�*
dtype0
�
1transformer_block/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*B
shared_name31transformer_block/multi_head_attention/key/kernel
�
Etransformer_block/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp1transformer_block/multi_head_attention/key/kernel*$
_output_shapes
:��*
dtype0
�
1transformer_block/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*B
shared_name31transformer_block/multi_head_attention/query/bias
�
Etransformer_block/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp1transformer_block/multi_head_attention/query/bias*
_output_shapes
:	�*
dtype0
�
3transformer_block/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*D
shared_name53transformer_block/multi_head_attention/query/kernel
�
Gtransformer_block/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp3transformer_block/multi_head_attention/query/kernel*$
_output_shapes
:��*
dtype0
�
3token_and_position_embedding/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	-�*D
shared_name53token_and_position_embedding/embedding_1/embeddings
�
Gtoken_and_position_embedding/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp3token_and_position_embedding/embedding_1/embeddings*
_output_shapes
:	-�*
dtype0
�
1token_and_position_embedding/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*B
shared_name31token_and_position_embedding/embedding/embeddings
�
Etoken_and_position_embedding/embedding/embeddings/Read/ReadVariableOpReadVariableOp1token_and_position_embedding/embedding/embeddings* 
_output_shapes
:
��*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:@*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:@*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	�@*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:�*
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
��*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:�*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�-�*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
�-�*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������-*
dtype0*
shape:���������-
�

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_13token_and_position_embedding/embedding_1/embeddings1token_and_position_embedding/embedding/embeddings3transformer_block/multi_head_attention/query/kernel1transformer_block/multi_head_attention/query/bias1transformer_block/multi_head_attention/key/kernel/transformer_block/multi_head_attention/key/bias3transformer_block/multi_head_attention/value/kernel1transformer_block/multi_head_attention/value/bias>transformer_block/multi_head_attention/attention_output/kernel<transformer_block/multi_head_attention/attention_output/bias+transformer_block/layer_normalization/gamma*transformer_block/layer_normalization/betadense/kernel
dense/biasdense_1/kerneldense_1/bias-transformer_block/layer_normalization_1/gamma,transformer_block/layer_normalization_1/betadense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*>
_read_only_resource_inputs 
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *+
f&R$
"__inference_signature_wrapper_3710

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*է
valueʧBƧ B��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	token_emb
pos_emb*
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%att
&ffn
'
layernorm1
(
layernorm2
)dropout1
*dropout2*
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses* 
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7_random_generator* 
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias*
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
F_random_generator* 
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

Mkernel
Nbias*
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
U_random_generator* 
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias*
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
d_random_generator* 
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias*
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
s_random_generator* 
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

zkernel
{bias*
�
|0
}1
~2
3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
>18
?19
M20
N21
\22
]23
k24
l25
z26
{27*
�
|0
}1
~2
3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
>18
?19
M20
N21
\22
]23
k24
l25
z26
{27*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 

�serving_default* 

|0
}1*

|0
}1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
|
embeddings*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
}
embeddings*
�
~0
1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
�
~0
1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense*
�
�layer_with_weights-0
�layer-0
�layer_with_weights-1
�layer-1
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

>0
?1*

>0
?1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

M0
N1*

M0
N1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

\0
]1*

\0
]1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

k0
l1*

k0
l1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

z0
{1*

z0
{1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1token_and_position_embedding/embedding/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE3token_and_position_embedding/embedding_1/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE3transformer_block/multi_head_attention/query/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1transformer_block/multi_head_attention/query/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1transformer_block/multi_head_attention/key/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/transformer_block/multi_head_attention/key/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE3transformer_block/multi_head_attention/value/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1transformer_block/multi_head_attention/value/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>transformer_block/multi_head_attention/attention_output/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE<transformer_block/multi_head_attention/attention_output/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
dense/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+transformer_block/layer_normalization/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*transformer_block/layer_normalization/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-transformer_block/layer_normalization_1/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,transformer_block/layer_normalization_1/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
* 
j
0
1
2
3
4
5
6
7
	8

9
10
11
12
13*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 

|0*

|0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

}0*

}0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
.
%0
&1
'2
(3
)4
*5*
* 
* 
* 
* 
* 
* 
* 
B
~0
1
�2
�3
�4
�5
�6
�7*
B
~0
1
�2
�3
�4
�5
�6
�7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

~kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 

~0
1*

~0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOpEtoken_and_position_embedding/embedding/embeddings/Read/ReadVariableOpGtoken_and_position_embedding/embedding_1/embeddings/Read/ReadVariableOpGtransformer_block/multi_head_attention/query/kernel/Read/ReadVariableOpEtransformer_block/multi_head_attention/query/bias/Read/ReadVariableOpEtransformer_block/multi_head_attention/key/kernel/Read/ReadVariableOpCtransformer_block/multi_head_attention/key/bias/Read/ReadVariableOpGtransformer_block/multi_head_attention/value/kernel/Read/ReadVariableOpEtransformer_block/multi_head_attention/value/bias/Read/ReadVariableOpRtransformer_block/multi_head_attention/attention_output/kernel/Read/ReadVariableOpPtransformer_block/multi_head_attention/attention_output/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp?transformer_block/layer_normalization/gamma/Read/ReadVariableOp>transformer_block/layer_normalization/beta/Read/ReadVariableOpAtransformer_block/layer_normalization_1/gamma/Read/ReadVariableOp@transformer_block/layer_normalization_1/beta/Read/ReadVariableOpConst*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *&
f!R
__inference__traced_save_6433
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/bias1token_and_position_embedding/embedding/embeddings3token_and_position_embedding/embedding_1/embeddings3transformer_block/multi_head_attention/query/kernel1transformer_block/multi_head_attention/query/bias1transformer_block/multi_head_attention/key/kernel/transformer_block/multi_head_attention/key/bias3transformer_block/multi_head_attention/value/kernel1transformer_block/multi_head_attention/value/bias>transformer_block/multi_head_attention/attention_output/kernel<transformer_block/multi_head_attention/attention_output/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias+transformer_block/layer_normalization/gamma*transformer_block/layer_normalization/beta-transformer_block/layer_normalization_1/gamma,transformer_block/layer_normalization_1/beta*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *)
f$R"
 __inference__traced_restore_6527��2
�
�
"__inference_signature_wrapper_3710
input_1
unknown:	-�
	unknown_0:
��!
	unknown_1:��
	unknown_2:	�!
	unknown_3:��
	unknown_4:	�!
	unknown_5:��
	unknown_6:	�!
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
�-�

unknown_18:	�

unknown_19:
��

unknown_20:	�

unknown_21:	�@

unknown_22:@

unknown_23:@

unknown_24:

unknown_25:

unknown_26:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*>
_read_only_resource_inputs 
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *(
f#R!
__inference__wrapped_model_1891s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������-: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������-
!
_user_specified_name	input_1
�

]
A__inference_reshape_layer_call_and_return_conditional_losses_5772

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :�-�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:����������-]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:����������-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������-�:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�
�
D__inference_sequential_layer_call_and_return_conditional_losses_2032

inputs

dense_2021:
��

dense_2023:	� 
dense_1_2026:
��
dense_1_2028:	�
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputs
dense_2021
dense_2023*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1929�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2026dense_1_2028*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1965|
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������-��
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������-�: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�_
�
?__inference_model_layer_call_and_return_conditional_losses_3506
input_14
!token_and_position_embedding_3368:	-�5
!token_and_position_embedding_3370:
��.
transformer_block_3373:��)
transformer_block_3375:	�.
transformer_block_3377:��)
transformer_block_3379:	�.
transformer_block_3381:��)
transformer_block_3383:	�.
transformer_block_3385:��%
transformer_block_3387:	�%
transformer_block_3389:	�%
transformer_block_3391:	�*
transformer_block_3393:
��%
transformer_block_3395:	�*
transformer_block_3397:
��%
transformer_block_3399:	�%
transformer_block_3401:	�%
transformer_block_3403:	� 
dense_2_3476:
�-�
dense_2_3478:	� 
dense_3_3482:
��
dense_3_3484:	�
dense_4_3488:	�@
dense_4_3490:@
dense_5_3494:@
dense_5_3496:
dense_6_3500:
dense_6_3502:
identity��dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�4token_and_position_embedding/StatefulPartitionedCall�)transformer_block/StatefulPartitionedCall�+transformer_block/StatefulPartitionedCall_1�+transformer_block/StatefulPartitionedCall_2�+transformer_block/StatefulPartitionedCall_3�+transformer_block/StatefulPartitionedCall_4�
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1!token_and_position_embedding_3368!token_and_position_embedding_3370*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *_
fZRX
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_2115�
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_block_3373transformer_block_3375transformer_block_3377transformer_block_3379transformer_block_3381transformer_block_3383transformer_block_3385transformer_block_3387transformer_block_3389transformer_block_3391transformer_block_3393transformer_block_3395transformer_block_3397transformer_block_3399transformer_block_3401transformer_block_3403*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2248�
+transformer_block/StatefulPartitionedCall_1StatefulPartitionedCall2transformer_block/StatefulPartitionedCall:output:0transformer_block_3373transformer_block_3375transformer_block_3377transformer_block_3379transformer_block_3381transformer_block_3383transformer_block_3385transformer_block_3387transformer_block_3389transformer_block_3391transformer_block_3393transformer_block_3395transformer_block_3397transformer_block_3399transformer_block_3401transformer_block_3403*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2248�
+transformer_block/StatefulPartitionedCall_2StatefulPartitionedCall4transformer_block/StatefulPartitionedCall_1:output:0transformer_block_3373transformer_block_3375transformer_block_3377transformer_block_3379transformer_block_3381transformer_block_3383transformer_block_3385transformer_block_3387transformer_block_3389transformer_block_3391transformer_block_3393transformer_block_3395transformer_block_3397transformer_block_3399transformer_block_3401transformer_block_3403*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2248�
+transformer_block/StatefulPartitionedCall_3StatefulPartitionedCall4transformer_block/StatefulPartitionedCall_2:output:0transformer_block_3373transformer_block_3375transformer_block_3377transformer_block_3379transformer_block_3381transformer_block_3383transformer_block_3385transformer_block_3387transformer_block_3389transformer_block_3391transformer_block_3393transformer_block_3395transformer_block_3397transformer_block_3399transformer_block_3401transformer_block_3403*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2248�
+transformer_block/StatefulPartitionedCall_4StatefulPartitionedCall4transformer_block/StatefulPartitionedCall_3:output:0transformer_block_3373transformer_block_3375transformer_block_3377transformer_block_3379transformer_block_3381transformer_block_3383transformer_block_3385transformer_block_3387transformer_block_3389transformer_block_3391transformer_block_3393transformer_block_3395transformer_block_3397transformer_block_3399transformer_block_3401transformer_block_3403*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2248�
reshape/PartitionedCallPartitionedCall4transformer_block/StatefulPartitionedCall_4:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_2363�
dropout_2/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_2370�
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_3476dense_2_3478*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_2403�
dropout_3/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_2414�
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_3_3482dense_3_3484*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_2447�
dropout_4/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_2458�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_4_3488dense_4_3490*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_2491�
dropout_5/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_5_layer_call_and_return_conditional_losses_2502�
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_5_3494dense_5_3496*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_2535�
dropout_6/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_6_layer_call_and_return_conditional_losses_2546�
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0dense_6_3500dense_6_3502*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_2579{
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall,^transformer_block/StatefulPartitionedCall_1,^transformer_block/StatefulPartitionedCall_2,^transformer_block/StatefulPartitionedCall_3,^transformer_block/StatefulPartitionedCall_4*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������-: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall2Z
+transformer_block/StatefulPartitionedCall_1+transformer_block/StatefulPartitionedCall_12Z
+transformer_block/StatefulPartitionedCall_2+transformer_block/StatefulPartitionedCall_22Z
+transformer_block/StatefulPartitionedCall_3+transformer_block/StatefulPartitionedCall_32Z
+transformer_block/StatefulPartitionedCall_4+transformer_block/StatefulPartitionedCall_4:P L
'
_output_shapes
:���������-
!
_user_specified_name	input_1
�
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_5787

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������-`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������-"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������-:T P
,
_output_shapes
:����������-
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_3365
input_1
unknown:	-�
	unknown_0:
��!
	unknown_1:��
	unknown_2:	�!
	unknown_3:��
	unknown_4:	�!
	unknown_5:��
	unknown_6:	�!
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
�-�

unknown_18:	�

unknown_19:
��

unknown_20:	�

unknown_21:	�@

unknown_22:@

unknown_23:@

unknown_24:

unknown_25:

unknown_26:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*>
_read_only_resource_inputs 
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_3245s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������-: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������-
!
_user_specified_name	input_1
�
D
(__inference_dropout_5_layer_call_fn_5978

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_5_layer_call_and_return_conditional_losses_2502d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
0__inference_transformer_block_layer_call_fn_5450

inputs
unknown:��
	unknown_0:	�!
	unknown_1:��
	unknown_2:	�!
	unknown_3:��
	unknown_4:	�!
	unknown_5:��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2248t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������-�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������-�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�
�
$__inference_dense_layer_call_fn_6256

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1929t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������-�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������-�: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�g
�
?__inference_model_layer_call_and_return_conditional_losses_3245

inputs4
!token_and_position_embedding_3107:	-�5
!token_and_position_embedding_3109:
��.
transformer_block_3112:��)
transformer_block_3114:	�.
transformer_block_3116:��)
transformer_block_3118:	�.
transformer_block_3120:��)
transformer_block_3122:	�.
transformer_block_3124:��%
transformer_block_3126:	�%
transformer_block_3128:	�%
transformer_block_3130:	�*
transformer_block_3132:
��%
transformer_block_3134:	�*
transformer_block_3136:
��%
transformer_block_3138:	�%
transformer_block_3140:	�%
transformer_block_3142:	� 
dense_2_3215:
�-�
dense_2_3217:	� 
dense_3_3221:
��
dense_3_3223:	�
dense_4_3227:	�@
dense_4_3229:@
dense_5_3233:@
dense_5_3235:
dense_6_3239:
dense_6_3241:
identity��dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�!dropout_3/StatefulPartitionedCall�!dropout_4/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCall�!dropout_6/StatefulPartitionedCall�4token_and_position_embedding/StatefulPartitionedCall�)transformer_block/StatefulPartitionedCall�+transformer_block/StatefulPartitionedCall_1�+transformer_block/StatefulPartitionedCall_2�+transformer_block/StatefulPartitionedCall_3�+transformer_block/StatefulPartitionedCall_4�
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs!token_and_position_embedding_3107!token_and_position_embedding_3109*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *_
fZRX
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_2115�
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_block_3112transformer_block_3114transformer_block_3116transformer_block_3118transformer_block_3120transformer_block_3122transformer_block_3124transformer_block_3126transformer_block_3128transformer_block_3130transformer_block_3132transformer_block_3134transformer_block_3136transformer_block_3138transformer_block_3140transformer_block_3142*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2996�
+transformer_block/StatefulPartitionedCall_1StatefulPartitionedCall2transformer_block/StatefulPartitionedCall:output:0transformer_block_3112transformer_block_3114transformer_block_3116transformer_block_3118transformer_block_3120transformer_block_3122transformer_block_3124transformer_block_3126transformer_block_3128transformer_block_3130transformer_block_3132transformer_block_3134transformer_block_3136transformer_block_3138transformer_block_3140transformer_block_3142*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2996�
+transformer_block/StatefulPartitionedCall_2StatefulPartitionedCall4transformer_block/StatefulPartitionedCall_1:output:0transformer_block_3112transformer_block_3114transformer_block_3116transformer_block_3118transformer_block_3120transformer_block_3122transformer_block_3124transformer_block_3126transformer_block_3128transformer_block_3130transformer_block_3132transformer_block_3134transformer_block_3136transformer_block_3138transformer_block_3140transformer_block_3142*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2996�
+transformer_block/StatefulPartitionedCall_3StatefulPartitionedCall4transformer_block/StatefulPartitionedCall_2:output:0transformer_block_3112transformer_block_3114transformer_block_3116transformer_block_3118transformer_block_3120transformer_block_3122transformer_block_3124transformer_block_3126transformer_block_3128transformer_block_3130transformer_block_3132transformer_block_3134transformer_block_3136transformer_block_3138transformer_block_3140transformer_block_3142*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2996�
+transformer_block/StatefulPartitionedCall_4StatefulPartitionedCall4transformer_block/StatefulPartitionedCall_3:output:0transformer_block_3112transformer_block_3114transformer_block_3116transformer_block_3118transformer_block_3120transformer_block_3122transformer_block_3124transformer_block_3126transformer_block_3128transformer_block_3130transformer_block_3132transformer_block_3134transformer_block_3136transformer_block_3138transformer_block_3140transformer_block_3142*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2996�
reshape/PartitionedCallPartitionedCall4transformer_block/StatefulPartitionedCall_4:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_2363�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_2807�
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_3215dense_2_3217*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_2403�
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_2774�
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_3_3221dense_3_3223*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_2447�
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_2741�
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_4_3227dense_4_3229*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_2491�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_5_layer_call_and_return_conditional_losses_2708�
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_5_3233dense_5_3235*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_2535�
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0"^dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_6_layer_call_and_return_conditional_losses_2675�
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0dense_6_3239dense_6_3241*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_2579{
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall,^transformer_block/StatefulPartitionedCall_1,^transformer_block/StatefulPartitionedCall_2,^transformer_block/StatefulPartitionedCall_3,^transformer_block/StatefulPartitionedCall_4*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������-: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall2Z
+transformer_block/StatefulPartitionedCall_1+transformer_block/StatefulPartitionedCall_12Z
+transformer_block/StatefulPartitionedCall_2+transformer_block/StatefulPartitionedCall_22Z
+transformer_block/StatefulPartitionedCall_3+transformer_block/StatefulPartitionedCall_32Z
+transformer_block/StatefulPartitionedCall_4+transformer_block/StatefulPartitionedCall_4:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�
�
D__inference_sequential_layer_call_and_return_conditional_losses_1972

inputs

dense_1930:
��

dense_1932:	� 
dense_1_1966:
��
dense_1_1968:	�
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputs
dense_1930
dense_1932*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1929�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_1966dense_1_1968*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1965|
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������-��
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������-�: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�

b
C__inference_dropout_6_layer_call_and_return_conditional_losses_6067

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * gL=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

b
C__inference_dropout_5_layer_call_and_return_conditional_losses_2708

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * gL=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������@s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������@m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������@]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
D
(__inference_dropout_6_layer_call_fn_6045

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_6_layer_call_and_return_conditional_losses_2546d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
A__inference_dense_3_layer_call_and_return_conditional_losses_5906

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
SeluSeluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentitySelu:activations:0^NoOp*
T0*,
_output_shapes
:����������z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_dense_6_layer_call_and_return_conditional_losses_2579

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
SeluSeluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentitySelu:activations:0^NoOp*
T0*+
_output_shapes
:���������z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_sequential_layer_call_and_return_conditional_losses_2084
dense_input

dense_2073:
��

dense_2075:	� 
dense_1_2078:
��
dense_1_2080:	�
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCalldense_input
dense_2073
dense_2075*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1929�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2078dense_1_2080*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1965|
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������-��
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������-�: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
,
_output_shapes
:���������-�
%
_user_specified_namedense_input
��
�
K__inference_transformer_block_layer_call_and_return_conditional_losses_2248

inputsX
@multi_head_attention_query_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_query_add_readvariableop_resource:	�V
>multi_head_attention_key_einsum_einsum_readvariableop_resource:��G
4multi_head_attention_key_add_readvariableop_resource:	�X
@multi_head_attention_value_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_value_add_readvariableop_resource:	�c
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:��P
Amulti_head_attention_attention_output_add_readvariableop_resource:	�H
9layer_normalization_batchnorm_mul_readvariableop_resource:	�D
5layer_normalization_batchnorm_readvariableop_resource:	�F
2sequential_dense_tensordot_readvariableop_resource:
��?
0sequential_dense_biasadd_readvariableop_resource:	�H
4sequential_dense_1_tensordot_readvariableop_resource:
��A
2sequential_dense_1_biasadd_readvariableop_resource:	�J
;layer_normalization_1_batchnorm_mul_readvariableop_resource:	�F
7layer_normalization_1_batchnorm_readvariableop_resource:	�
identity��,layer_normalization/batchnorm/ReadVariableOp�0layer_normalization/batchnorm/mul/ReadVariableOp�.layer_normalization_1/batchnorm/ReadVariableOp�2layer_normalization_1/batchnorm/mul/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOp�'sequential/dense/BiasAdd/ReadVariableOp�)sequential/dense/Tensordot/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�+sequential/dense_1/Tensordot/ReadVariableOp�
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:���������-��
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������--�
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������--�
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
dropout/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:���������-�f
addAddV2inputsdropout/Identity:output:0*
T0*,
_output_shapes
:���������-�|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:���������-�
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������-�
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:���������-�
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������-��
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������-��
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       w
 sequential/dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$sequential/dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0*sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-��
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�w
sequential/dense/SeluSelu!sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:���������-��
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0k
!sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       u
"sequential/dense_1/Tensordot/ShapeShape#sequential/dense/Selu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense_1/Tensordot/GatherV2GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/free:output:03sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential/dense_1/Tensordot/GatherV2_1GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/axes:output:05sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
&sequential/dense_1/Tensordot/transpose	Transpose#sequential/dense/Selu:activations:0,sequential/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������o
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-��
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�z
dropout_1/IdentityIdentity#sequential/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������-��
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*,
_output_shapes
:���������-�~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:���������-�
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������-�
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:���������-�
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������-��
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������-�}
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������-��
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������-�: : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�
D
(__inference_dropout_2_layer_call_fn_5777

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_2370e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������-:T P
,
_output_shapes
:����������-
 
_user_specified_nameinputs
�
�
&__inference_dense_3_layer_call_fn_5875

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_2447t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

b
C__inference_dropout_3_layer_call_and_return_conditional_losses_5866

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * gL=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�_
�
?__inference_model_layer_call_and_return_conditional_losses_2586

inputs4
!token_and_position_embedding_2116:	-�5
!token_and_position_embedding_2118:
��.
transformer_block_2249:��)
transformer_block_2251:	�.
transformer_block_2253:��)
transformer_block_2255:	�.
transformer_block_2257:��)
transformer_block_2259:	�.
transformer_block_2261:��%
transformer_block_2263:	�%
transformer_block_2265:	�%
transformer_block_2267:	�*
transformer_block_2269:
��%
transformer_block_2271:	�*
transformer_block_2273:
��%
transformer_block_2275:	�%
transformer_block_2277:	�%
transformer_block_2279:	� 
dense_2_2404:
�-�
dense_2_2406:	� 
dense_3_2448:
��
dense_3_2450:	�
dense_4_2492:	�@
dense_4_2494:@
dense_5_2536:@
dense_5_2538:
dense_6_2580:
dense_6_2582:
identity��dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�4token_and_position_embedding/StatefulPartitionedCall�)transformer_block/StatefulPartitionedCall�+transformer_block/StatefulPartitionedCall_1�+transformer_block/StatefulPartitionedCall_2�+transformer_block/StatefulPartitionedCall_3�+transformer_block/StatefulPartitionedCall_4�
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs!token_and_position_embedding_2116!token_and_position_embedding_2118*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *_
fZRX
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_2115�
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_block_2249transformer_block_2251transformer_block_2253transformer_block_2255transformer_block_2257transformer_block_2259transformer_block_2261transformer_block_2263transformer_block_2265transformer_block_2267transformer_block_2269transformer_block_2271transformer_block_2273transformer_block_2275transformer_block_2277transformer_block_2279*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2248�
+transformer_block/StatefulPartitionedCall_1StatefulPartitionedCall2transformer_block/StatefulPartitionedCall:output:0transformer_block_2249transformer_block_2251transformer_block_2253transformer_block_2255transformer_block_2257transformer_block_2259transformer_block_2261transformer_block_2263transformer_block_2265transformer_block_2267transformer_block_2269transformer_block_2271transformer_block_2273transformer_block_2275transformer_block_2277transformer_block_2279*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2248�
+transformer_block/StatefulPartitionedCall_2StatefulPartitionedCall4transformer_block/StatefulPartitionedCall_1:output:0transformer_block_2249transformer_block_2251transformer_block_2253transformer_block_2255transformer_block_2257transformer_block_2259transformer_block_2261transformer_block_2263transformer_block_2265transformer_block_2267transformer_block_2269transformer_block_2271transformer_block_2273transformer_block_2275transformer_block_2277transformer_block_2279*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2248�
+transformer_block/StatefulPartitionedCall_3StatefulPartitionedCall4transformer_block/StatefulPartitionedCall_2:output:0transformer_block_2249transformer_block_2251transformer_block_2253transformer_block_2255transformer_block_2257transformer_block_2259transformer_block_2261transformer_block_2263transformer_block_2265transformer_block_2267transformer_block_2269transformer_block_2271transformer_block_2273transformer_block_2275transformer_block_2277transformer_block_2279*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2248�
+transformer_block/StatefulPartitionedCall_4StatefulPartitionedCall4transformer_block/StatefulPartitionedCall_3:output:0transformer_block_2249transformer_block_2251transformer_block_2253transformer_block_2255transformer_block_2257transformer_block_2259transformer_block_2261transformer_block_2263transformer_block_2265transformer_block_2267transformer_block_2269transformer_block_2271transformer_block_2273transformer_block_2275transformer_block_2277transformer_block_2279*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2248�
reshape/PartitionedCallPartitionedCall4transformer_block/StatefulPartitionedCall_4:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_2363�
dropout_2/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_2370�
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_2404dense_2_2406*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_2403�
dropout_3/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_2414�
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_3_2448dense_3_2450*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_2447�
dropout_4/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_2458�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_4_2492dense_4_2494*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_2491�
dropout_5/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_5_layer_call_and_return_conditional_losses_2502�
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_5_2536dense_5_2538*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_2535�
dropout_6/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_6_layer_call_and_return_conditional_losses_2546�
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0dense_6_2580dense_6_2582*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_2579{
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall,^transformer_block/StatefulPartitionedCall_1,^transformer_block/StatefulPartitionedCall_2,^transformer_block/StatefulPartitionedCall_3,^transformer_block/StatefulPartitionedCall_4*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������-: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall2Z
+transformer_block/StatefulPartitionedCall_1+transformer_block/StatefulPartitionedCall_12Z
+transformer_block/StatefulPartitionedCall_2+transformer_block/StatefulPartitionedCall_22Z
+transformer_block/StatefulPartitionedCall_3+transformer_block/StatefulPartitionedCall_32Z
+transformer_block/StatefulPartitionedCall_4+transformer_block/StatefulPartitionedCall_4:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�
�
A__inference_dense_4_layer_call_and_return_conditional_losses_5973

inputs4
!tensordot_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@T
SeluSeluBiasAdd:output:0*
T0*+
_output_shapes
:���������@e
IdentityIdentitySelu:activations:0^NoOp*
T0*+
_output_shapes
:���������@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
?__inference_dense_layer_call_and_return_conditional_losses_6287

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-�s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�U
SeluSeluBiasAdd:output:0*
T0*,
_output_shapes
:���������-�f
IdentityIdentitySelu:activations:0^NoOp*
T0*,
_output_shapes
:���������-�z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������-�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�
D
(__inference_dropout_4_layer_call_fn_5911

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_2458e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

b
C__inference_dropout_4_layer_call_and_return_conditional_losses_5933

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * gL=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
D
(__inference_dropout_3_layer_call_fn_5844

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_2414e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_sequential_layer_call_and_return_conditional_losses_2070
dense_input

dense_2059:
��

dense_2061:	� 
dense_1_2064:
��
dense_1_2066:	�
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCalldense_input
dense_2059
dense_2061*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1929�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2064dense_1_2066*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1965|
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������-��
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������-�: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
,
_output_shapes
:���������-�
%
_user_specified_namedense_input
�
�
)__inference_sequential_layer_call_fn_2056
dense_input
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_2032t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������-�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������-�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:���������-�
%
_user_specified_namedense_input
��
�
K__inference_transformer_block_layer_call_and_return_conditional_losses_5614

inputsX
@multi_head_attention_query_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_query_add_readvariableop_resource:	�V
>multi_head_attention_key_einsum_einsum_readvariableop_resource:��G
4multi_head_attention_key_add_readvariableop_resource:	�X
@multi_head_attention_value_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_value_add_readvariableop_resource:	�c
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:��P
Amulti_head_attention_attention_output_add_readvariableop_resource:	�H
9layer_normalization_batchnorm_mul_readvariableop_resource:	�D
5layer_normalization_batchnorm_readvariableop_resource:	�F
2sequential_dense_tensordot_readvariableop_resource:
��?
0sequential_dense_biasadd_readvariableop_resource:	�H
4sequential_dense_1_tensordot_readvariableop_resource:
��A
2sequential_dense_1_biasadd_readvariableop_resource:	�J
;layer_normalization_1_batchnorm_mul_readvariableop_resource:	�F
7layer_normalization_1_batchnorm_readvariableop_resource:	�
identity��,layer_normalization/batchnorm/ReadVariableOp�0layer_normalization/batchnorm/mul/ReadVariableOp�.layer_normalization_1/batchnorm/ReadVariableOp�2layer_normalization_1/batchnorm/mul/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOp�'sequential/dense/BiasAdd/ReadVariableOp�)sequential/dense/Tensordot/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�+sequential/dense_1/Tensordot/ReadVariableOp�
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:���������-��
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������--�
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������--�
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
dropout/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:���������-�f
addAddV2inputsdropout/Identity:output:0*
T0*,
_output_shapes
:���������-�|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:���������-�
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������-�
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:���������-�
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������-��
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������-��
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       w
 sequential/dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$sequential/dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0*sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-��
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�w
sequential/dense/SeluSelu!sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:���������-��
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0k
!sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       u
"sequential/dense_1/Tensordot/ShapeShape#sequential/dense/Selu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense_1/Tensordot/GatherV2GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/free:output:03sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential/dense_1/Tensordot/GatherV2_1GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/axes:output:05sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
&sequential/dense_1/Tensordot/transpose	Transpose#sequential/dense/Selu:activations:0,sequential/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������o
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-��
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�z
dropout_1/IdentityIdentity#sequential/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������-��
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*,
_output_shapes
:���������-�~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:���������-�
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������-�
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:���������-�
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������-��
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������-�}
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������-��
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������-�: : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�
�
;__inference_token_and_position_embedding_layer_call_fn_5389
x
unknown:	-�
	unknown_0:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *_
fZRX
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_2115t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������-�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:���������-

_user_specified_namex
�
B
&__inference_reshape_layer_call_fn_5759

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_2363e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������-�:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�
�
)__inference_sequential_layer_call_fn_6133

inputs
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_2032t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������-�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������-�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�
�
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_2115
x4
!embedding_1_embedding_lookup_2102:	-�3
embedding_embedding_lookup_2108:
��
identity��embedding/embedding_lookup�embedding_1/embedding_lookup6
ShapeShapex*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*
_output_shapes
:-�
embedding_1/embedding_lookupResourceGather!embedding_1_embedding_lookup_2102range:output:0*
Tindices0*4
_class*
(&loc:@embedding_1/embedding_lookup/2102*
_output_shapes
:	-�*
dtype0�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding_1/embedding_lookup/2102*
_output_shapes
:	-��
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	-�Z
embedding/CastCastx*

DstT0*

SrcT0*'
_output_shapes
:���������-�
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_2108embedding/Cast:y:0*
Tindices0*2
_class(
&$loc:@embedding/embedding_lookup/2108*,
_output_shapes
:���������-�*
dtype0�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/2108*,
_output_shapes
:���������-��
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:���������-��
addAddV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:���������-�[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:���������-��
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:J F
'
_output_shapes
:���������-

_user_specified_namex
�
�
&__inference_dense_5_layer_call_fn_6009

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_2535s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�

]
A__inference_reshape_layer_call_and_return_conditional_losses_2363

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :�-�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:����������-]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:����������-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������-�:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
̎

�B
?__inference_model_layer_call_and_return_conditional_losses_4556

inputsQ
>token_and_position_embedding_embedding_1_embedding_lookup_3843:	-�P
<token_and_position_embedding_embedding_embedding_lookup_3849:
��j
Rtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource:��[
Htransformer_block_multi_head_attention_query_add_readvariableop_resource:	�h
Ptransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource:��Y
Ftransformer_block_multi_head_attention_key_add_readvariableop_resource:	�j
Rtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource:��[
Htransformer_block_multi_head_attention_value_add_readvariableop_resource:	�u
]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:��b
Stransformer_block_multi_head_attention_attention_output_add_readvariableop_resource:	�Z
Ktransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource:	�V
Gtransformer_block_layer_normalization_batchnorm_readvariableop_resource:	�X
Dtransformer_block_sequential_dense_tensordot_readvariableop_resource:
��Q
Btransformer_block_sequential_dense_biasadd_readvariableop_resource:	�Z
Ftransformer_block_sequential_dense_1_tensordot_readvariableop_resource:
��S
Dtransformer_block_sequential_dense_1_biasadd_readvariableop_resource:	�\
Mtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource:	�X
Itransformer_block_layer_normalization_1_batchnorm_readvariableop_resource:	�=
)dense_2_tensordot_readvariableop_resource:
�-�6
'dense_2_biasadd_readvariableop_resource:	�=
)dense_3_tensordot_readvariableop_resource:
��6
'dense_3_biasadd_readvariableop_resource:	�<
)dense_4_tensordot_readvariableop_resource:	�@5
'dense_4_biasadd_readvariableop_resource:@;
)dense_5_tensordot_readvariableop_resource:@5
'dense_5_biasadd_readvariableop_resource:;
)dense_6_tensordot_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:
identity��dense_2/BiasAdd/ReadVariableOp� dense_2/Tensordot/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp� dense_3/Tensordot/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp� dense_4/Tensordot/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp� dense_5/Tensordot/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp� dense_6/Tensordot/ReadVariableOp�7token_and_position_embedding/embedding/embedding_lookup�9token_and_position_embedding/embedding_1/embedding_lookup�>transformer_block/layer_normalization/batchnorm/ReadVariableOp�Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp�@transformer_block/layer_normalization/batchnorm_1/ReadVariableOp�Dtransformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOp�@transformer_block/layer_normalization/batchnorm_2/ReadVariableOp�Dtransformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOp�@transformer_block/layer_normalization/batchnorm_3/ReadVariableOp�Dtransformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOp�@transformer_block/layer_normalization/batchnorm_4/ReadVariableOp�Dtransformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOp�@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp�Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp�Btransformer_block/layer_normalization_1/batchnorm_1/ReadVariableOp�Ftransformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOp�Btransformer_block/layer_normalization_1/batchnorm_2/ReadVariableOp�Ftransformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOp�Btransformer_block/layer_normalization_1/batchnorm_3/ReadVariableOp�Ftransformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOp�Btransformer_block/layer_normalization_1/batchnorm_4/ReadVariableOp�Ftransformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOp�Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp�Ltransformer_block/multi_head_attention/attention_output/add_1/ReadVariableOp�Ltransformer_block/multi_head_attention/attention_output/add_2/ReadVariableOp�Ltransformer_block/multi_head_attention/attention_output/add_3/ReadVariableOp�Ltransformer_block/multi_head_attention/attention_output/add_4/ReadVariableOp�Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp�Vtransformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOp�Vtransformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOp�Vtransformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOp�Vtransformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOp�=transformer_block/multi_head_attention/key/add/ReadVariableOp�?transformer_block/multi_head_attention/key/add_1/ReadVariableOp�?transformer_block/multi_head_attention/key/add_2/ReadVariableOp�?transformer_block/multi_head_attention/key/add_3/ReadVariableOp�?transformer_block/multi_head_attention/key/add_4/ReadVariableOp�Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp�Itransformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOp�Itransformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOp�Itransformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOp�Itransformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOp�?transformer_block/multi_head_attention/query/add/ReadVariableOp�Atransformer_block/multi_head_attention/query/add_1/ReadVariableOp�Atransformer_block/multi_head_attention/query/add_2/ReadVariableOp�Atransformer_block/multi_head_attention/query/add_3/ReadVariableOp�Atransformer_block/multi_head_attention/query/add_4/ReadVariableOp�Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp�Ktransformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOp�Ktransformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOp�Ktransformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOp�Ktransformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOp�?transformer_block/multi_head_attention/value/add/ReadVariableOp�Atransformer_block/multi_head_attention/value/add_1/ReadVariableOp�Atransformer_block/multi_head_attention/value/add_2/ReadVariableOp�Atransformer_block/multi_head_attention/value/add_3/ReadVariableOp�Atransformer_block/multi_head_attention/value/add_4/ReadVariableOp�Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp�Ktransformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOp�Ktransformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOp�Ktransformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOp�Ktransformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOp�9transformer_block/sequential/dense/BiasAdd/ReadVariableOp�;transformer_block/sequential/dense/BiasAdd_1/ReadVariableOp�;transformer_block/sequential/dense/BiasAdd_2/ReadVariableOp�;transformer_block/sequential/dense/BiasAdd_3/ReadVariableOp�;transformer_block/sequential/dense/BiasAdd_4/ReadVariableOp�;transformer_block/sequential/dense/Tensordot/ReadVariableOp�=transformer_block/sequential/dense/Tensordot_1/ReadVariableOp�=transformer_block/sequential/dense/Tensordot_2/ReadVariableOp�=transformer_block/sequential/dense/Tensordot_3/ReadVariableOp�=transformer_block/sequential/dense/Tensordot_4/ReadVariableOp�;transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp�=transformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOp�=transformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOp�=transformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOp�=transformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOp�=transformer_block/sequential/dense_1/Tensordot/ReadVariableOp�?transformer_block/sequential/dense_1/Tensordot_1/ReadVariableOp�?transformer_block/sequential/dense_1/Tensordot_2/ReadVariableOp�?transformer_block/sequential/dense_1/Tensordot_3/ReadVariableOp�?transformer_block/sequential/dense_1/Tensordot_4/ReadVariableOpX
"token_and_position_embedding/ShapeShapeinputs*
T0*
_output_shapes
:�
0token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������|
2token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*token_and_position_embedding/strided_sliceStridedSlice+token_and_position_embedding/Shape:output:09token_and_position_embedding/strided_slice/stack:output:0;token_and_position_embedding/strided_slice/stack_1:output:0;token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : j
(token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
"token_and_position_embedding/rangeRange1token_and_position_embedding/range/start:output:03token_and_position_embedding/strided_slice:output:01token_and_position_embedding/range/delta:output:0*
_output_shapes
:-�
9token_and_position_embedding/embedding_1/embedding_lookupResourceGather>token_and_position_embedding_embedding_1_embedding_lookup_3843+token_and_position_embedding/range:output:0*
Tindices0*Q
_classG
ECloc:@token_and_position_embedding/embedding_1/embedding_lookup/3843*
_output_shapes
:	-�*
dtype0�
Btoken_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityBtoken_and_position_embedding/embedding_1/embedding_lookup:output:0*
T0*Q
_classG
ECloc:@token_and_position_embedding/embedding_1/embedding_lookup/3843*
_output_shapes
:	-��
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityKtoken_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	-�|
+token_and_position_embedding/embedding/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������-�
7token_and_position_embedding/embedding/embedding_lookupResourceGather<token_and_position_embedding_embedding_embedding_lookup_3849/token_and_position_embedding/embedding/Cast:y:0*
Tindices0*O
_classE
CAloc:@token_and_position_embedding/embedding/embedding_lookup/3849*,
_output_shapes
:���������-�*
dtype0�
@token_and_position_embedding/embedding/embedding_lookup/IdentityIdentity@token_and_position_embedding/embedding/embedding_lookup:output:0*
T0*O
_classE
CAloc:@token_and_position_embedding/embedding/embedding_lookup/3849*,
_output_shapes
:���������-��
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityItoken_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:���������-��
 token_and_position_embedding/addAddV2Ktoken_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Mtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:���������-��
Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
:transformer_block/multi_head_attention/query/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Qtransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
?transformer_block/multi_head_attention/query/add/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
0transformer_block/multi_head_attention/query/addAddV2Ctransformer_block/multi_head_attention/query/einsum/Einsum:output:0Gtransformer_block/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpPtransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
8transformer_block/multi_head_attention/key/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Otransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
=transformer_block/multi_head_attention/key/add/ReadVariableOpReadVariableOpFtransformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
.transformer_block/multi_head_attention/key/addAddV2Atransformer_block/multi_head_attention/key/einsum/Einsum:output:0Etransformer_block/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
:transformer_block/multi_head_attention/value/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Qtransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
?transformer_block/multi_head_attention/value/add/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
0transformer_block/multi_head_attention/value/addAddV2Ctransformer_block/multi_head_attention/value/einsum/Einsum:output:0Gtransformer_block/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�q
,transformer_block/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
*transformer_block/multi_head_attention/MulMul4transformer_block/multi_head_attention/query/add:z:05transformer_block/multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:���������-��
4transformer_block/multi_head_attention/einsum/EinsumEinsum2transformer_block/multi_head_attention/key/add:z:0.transformer_block/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
6transformer_block/multi_head_attention/softmax/SoftmaxSoftmax=transformer_block/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������--�
7transformer_block/multi_head_attention/dropout/IdentityIdentity@transformer_block/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������--�
6transformer_block/multi_head_attention/einsum_1/EinsumEinsum@transformer_block/multi_head_attention/dropout/Identity:output:04transformer_block/multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Etransformer_block/multi_head_attention/attention_output/einsum/EinsumEinsum?transformer_block/multi_head_attention/einsum_1/Einsum:output:0\transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpStransformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;transformer_block/multi_head_attention/attention_output/addAddV2Ntransformer_block/multi_head_attention/attention_output/einsum/Einsum:output:0Rtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
"transformer_block/dropout/IdentityIdentity?transformer_block/multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:���������-��
transformer_block/addAddV2$token_and_position_embedding/add:z:0+transformer_block/dropout/Identity:output:0*
T0*,
_output_shapes
:���������-��
Dtransformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
2transformer_block/layer_normalization/moments/meanMeantransformer_block/add:z:0Mtransformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
:transformer_block/layer_normalization/moments/StopGradientStopGradient;transformer_block/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:���������-�
?transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_block/add:z:0Ctransformer_block/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Htransformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
6transformer_block/layer_normalization/moments/varianceMeanCtransformer_block/layer_normalization/moments/SquaredDifference:z:0Qtransformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(z
5transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
3transformer_block/layer_normalization/batchnorm/addAddV2?transformer_block/layer_normalization/moments/variance:output:0>transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������-�
5transformer_block/layer_normalization/batchnorm/RsqrtRsqrt7transformer_block/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:���������-�
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
3transformer_block/layer_normalization/batchnorm/mulMul9transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Jtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
5transformer_block/layer_normalization/batchnorm/mul_1Multransformer_block/add:z:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
5transformer_block/layer_normalization/batchnorm/mul_2Mul;transformer_block/layer_normalization/moments/mean:output:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
>transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
3transformer_block/layer_normalization/batchnorm/subSubFtransformer_block/layer_normalization/batchnorm/ReadVariableOp:value:09transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������-��
5transformer_block/layer_normalization/batchnorm/add_1AddV29transformer_block/layer_normalization/batchnorm/mul_1:z:07transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������-��
;transformer_block/sequential/dense/Tensordot/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0{
1transformer_block/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
1transformer_block/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
2transformer_block/sequential/dense/Tensordot/ShapeShape9transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:|
:transformer_block/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5transformer_block/sequential/dense/Tensordot/GatherV2GatherV2;transformer_block/sequential/dense/Tensordot/Shape:output:0:transformer_block/sequential/dense/Tensordot/free:output:0Ctransformer_block/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
<transformer_block/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot/GatherV2_1GatherV2;transformer_block/sequential/dense/Tensordot/Shape:output:0:transformer_block/sequential/dense/Tensordot/axes:output:0Etransformer_block/sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:|
2transformer_block/sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
1transformer_block/sequential/dense/Tensordot/ProdProd>transformer_block/sequential/dense/Tensordot/GatherV2:output:0;transformer_block/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: ~
4transformer_block/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
3transformer_block/sequential/dense/Tensordot/Prod_1Prod@transformer_block/sequential/dense/Tensordot/GatherV2_1:output:0=transformer_block/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: z
8transformer_block/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
3transformer_block/sequential/dense/Tensordot/concatConcatV2:transformer_block/sequential/dense/Tensordot/free:output:0:transformer_block/sequential/dense/Tensordot/axes:output:0Atransformer_block/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
2transformer_block/sequential/dense/Tensordot/stackPack:transformer_block/sequential/dense/Tensordot/Prod:output:0<transformer_block/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
6transformer_block/sequential/dense/Tensordot/transpose	Transpose9transformer_block/layer_normalization/batchnorm/add_1:z:0<transformer_block/sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
4transformer_block/sequential/dense/Tensordot/ReshapeReshape:transformer_block/sequential/dense/Tensordot/transpose:y:0;transformer_block/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
3transformer_block/sequential/dense/Tensordot/MatMulMatMul=transformer_block/sequential/dense/Tensordot/Reshape:output:0Ctransformer_block/sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
4transformer_block/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�|
:transformer_block/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5transformer_block/sequential/dense/Tensordot/concat_1ConcatV2>transformer_block/sequential/dense/Tensordot/GatherV2:output:0=transformer_block/sequential/dense/Tensordot/Const_2:output:0Ctransformer_block/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
,transformer_block/sequential/dense/TensordotReshape=transformer_block/sequential/dense/Tensordot/MatMul:product:0>transformer_block/sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-��
9transformer_block/sequential/dense/BiasAdd/ReadVariableOpReadVariableOpBtransformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*transformer_block/sequential/dense/BiasAddBiasAdd5transformer_block/sequential/dense/Tensordot:output:0Atransformer_block/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
'transformer_block/sequential/dense/SeluSelu3transformer_block/sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense_1/Tensordot/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0}
3transformer_block/sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
3transformer_block/sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
4transformer_block/sequential/dense_1/Tensordot/ShapeShape5transformer_block/sequential/dense/Selu:activations:0*
T0*
_output_shapes
:~
<transformer_block/sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense_1/Tensordot/GatherV2GatherV2=transformer_block/sequential/dense_1/Tensordot/Shape:output:0<transformer_block/sequential/dense_1/Tensordot/free:output:0Etransformer_block/sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
>transformer_block/sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot/GatherV2_1GatherV2=transformer_block/sequential/dense_1/Tensordot/Shape:output:0<transformer_block/sequential/dense_1/Tensordot/axes:output:0Gtransformer_block/sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4transformer_block/sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
3transformer_block/sequential/dense_1/Tensordot/ProdProd@transformer_block/sequential/dense_1/Tensordot/GatherV2:output:0=transformer_block/sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
6transformer_block/sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense_1/Tensordot/Prod_1ProdBtransformer_block/sequential/dense_1/Tensordot/GatherV2_1:output:0?transformer_block/sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_block/sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5transformer_block/sequential/dense_1/Tensordot/concatConcatV2<transformer_block/sequential/dense_1/Tensordot/free:output:0<transformer_block/sequential/dense_1/Tensordot/axes:output:0Ctransformer_block/sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
4transformer_block/sequential/dense_1/Tensordot/stackPack<transformer_block/sequential/dense_1/Tensordot/Prod:output:0>transformer_block/sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
8transformer_block/sequential/dense_1/Tensordot/transpose	Transpose5transformer_block/sequential/dense/Selu:activations:0>transformer_block/sequential/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
6transformer_block/sequential/dense_1/Tensordot/ReshapeReshape<transformer_block/sequential/dense_1/Tensordot/transpose:y:0=transformer_block/sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
5transformer_block/sequential/dense_1/Tensordot/MatMulMatMul?transformer_block/sequential/dense_1/Tensordot/Reshape:output:0Etransformer_block/sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
6transformer_block/sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�~
<transformer_block/sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense_1/Tensordot/concat_1ConcatV2@transformer_block/sequential/dense_1/Tensordot/GatherV2:output:0?transformer_block/sequential/dense_1/Tensordot/Const_2:output:0Etransformer_block/sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
.transformer_block/sequential/dense_1/TensordotReshape?transformer_block/sequential/dense_1/Tensordot/MatMul:product:0@transformer_block/sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-��
;transformer_block/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,transformer_block/sequential/dense_1/BiasAddBiasAdd7transformer_block/sequential/dense_1/Tensordot:output:0Ctransformer_block/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
$transformer_block/dropout_1/IdentityIdentity5transformer_block/sequential/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_1AddV29transformer_block/layer_normalization/batchnorm/add_1:z:0-transformer_block/dropout_1/Identity:output:0*
T0*,
_output_shapes
:���������-��
Ftransformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
4transformer_block/layer_normalization_1/moments/meanMeantransformer_block/add_1:z:0Otransformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
<transformer_block/layer_normalization_1/moments/StopGradientStopGradient=transformer_block/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:���������-�
Atransformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_block/add_1:z:0Etransformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Jtransformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
8transformer_block/layer_normalization_1/moments/varianceMeanEtransformer_block/layer_normalization_1/moments/SquaredDifference:z:0Stransformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(|
7transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
5transformer_block/layer_normalization_1/batchnorm/addAddV2Atransformer_block/layer_normalization_1/moments/variance:output:0@transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������-�
7transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt9transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:���������-�
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization_1/batchnorm/mulMul;transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Ltransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization_1/batchnorm/mul_1Multransformer_block/add_1:z:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization_1/batchnorm/mul_2Mul=transformer_block/layer_normalization_1/moments/mean:output:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization_1/batchnorm/subSubHtransformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0;transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization_1/batchnorm/add_1AddV2;transformer_block/layer_normalization_1/batchnorm/mul_1:z:09transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������-��
Ktransformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_block/multi_head_attention/query/einsum_1/EinsumEinsum;transformer_block/layer_normalization_1/batchnorm/add_1:z:0Stransformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Atransformer_block/multi_head_attention/query/add_1/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_block/multi_head_attention/query/add_1AddV2Etransformer_block/multi_head_attention/query/einsum_1/Einsum:output:0Itransformer_block/multi_head_attention/query/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Itransformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOpReadVariableOpPtransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
:transformer_block/multi_head_attention/key/einsum_1/EinsumEinsum;transformer_block/layer_normalization_1/batchnorm/add_1:z:0Qtransformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
?transformer_block/multi_head_attention/key/add_1/ReadVariableOpReadVariableOpFtransformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
0transformer_block/multi_head_attention/key/add_1AddV2Ctransformer_block/multi_head_attention/key/einsum_1/Einsum:output:0Gtransformer_block/multi_head_attention/key/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Ktransformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_block/multi_head_attention/value/einsum_1/EinsumEinsum;transformer_block/layer_normalization_1/batchnorm/add_1:z:0Stransformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Atransformer_block/multi_head_attention/value/add_1/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_block/multi_head_attention/value/add_1AddV2Etransformer_block/multi_head_attention/value/einsum_1/Einsum:output:0Itransformer_block/multi_head_attention/value/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�s
.transformer_block/multi_head_attention/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
,transformer_block/multi_head_attention/Mul_1Mul6transformer_block/multi_head_attention/query/add_1:z:07transformer_block/multi_head_attention/Mul_1/y:output:0*
T0*0
_output_shapes
:���������-��
6transformer_block/multi_head_attention/einsum_2/EinsumEinsum4transformer_block/multi_head_attention/key/add_1:z:00transformer_block/multi_head_attention/Mul_1:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
8transformer_block/multi_head_attention/softmax/Softmax_1Softmax?transformer_block/multi_head_attention/einsum_2/Einsum:output:0*
T0*/
_output_shapes
:���������--�
9transformer_block/multi_head_attention/dropout/Identity_1IdentityBtransformer_block/multi_head_attention/softmax/Softmax_1:softmax:0*
T0*/
_output_shapes
:���������--�
6transformer_block/multi_head_attention/einsum_3/EinsumEinsumBtransformer_block/multi_head_attention/dropout/Identity_1:output:06transformer_block/multi_head_attention/value/add_1:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
Vtransformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOpReadVariableOp]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Gtransformer_block/multi_head_attention/attention_output/einsum_1/EinsumEinsum?transformer_block/multi_head_attention/einsum_3/Einsum:output:0^transformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
Ltransformer_block/multi_head_attention/attention_output/add_1/ReadVariableOpReadVariableOpStransformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=transformer_block/multi_head_attention/attention_output/add_1AddV2Ptransformer_block/multi_head_attention/attention_output/einsum_1/Einsum:output:0Ttransformer_block/multi_head_attention/attention_output/add_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
$transformer_block/dropout/Identity_1IdentityAtransformer_block/multi_head_attention/attention_output/add_1:z:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_2AddV2;transformer_block/layer_normalization_1/batchnorm/add_1:z:0-transformer_block/dropout/Identity_1:output:0*
T0*,
_output_shapes
:���������-��
Ftransformer_block/layer_normalization/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
4transformer_block/layer_normalization/moments_1/meanMeantransformer_block/add_2:z:0Otransformer_block/layer_normalization/moments_1/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
<transformer_block/layer_normalization/moments_1/StopGradientStopGradient=transformer_block/layer_normalization/moments_1/mean:output:0*
T0*+
_output_shapes
:���������-�
Atransformer_block/layer_normalization/moments_1/SquaredDifferenceSquaredDifferencetransformer_block/add_2:z:0Etransformer_block/layer_normalization/moments_1/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Jtransformer_block/layer_normalization/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
8transformer_block/layer_normalization/moments_1/varianceMeanEtransformer_block/layer_normalization/moments_1/SquaredDifference:z:0Stransformer_block/layer_normalization/moments_1/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(|
7transformer_block/layer_normalization/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
5transformer_block/layer_normalization/batchnorm_1/addAddV2Atransformer_block/layer_normalization/moments_1/variance:output:0@transformer_block/layer_normalization/batchnorm_1/add/y:output:0*
T0*+
_output_shapes
:���������-�
7transformer_block/layer_normalization/batchnorm_1/RsqrtRsqrt9transformer_block/layer_normalization/batchnorm_1/add:z:0*
T0*+
_output_shapes
:���������-�
Dtransformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization/batchnorm_1/mulMul;transformer_block/layer_normalization/batchnorm_1/Rsqrt:y:0Ltransformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_1/mul_1Multransformer_block/add_2:z:09transformer_block/layer_normalization/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_1/mul_2Mul=transformer_block/layer_normalization/moments_1/mean:output:09transformer_block/layer_normalization/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:���������-��
@transformer_block/layer_normalization/batchnorm_1/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization/batchnorm_1/subSubHtransformer_block/layer_normalization/batchnorm_1/ReadVariableOp:value:0;transformer_block/layer_normalization/batchnorm_1/mul_2:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_1/add_1AddV2;transformer_block/layer_normalization/batchnorm_1/mul_1:z:09transformer_block/layer_normalization/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense/Tensordot_1/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0}
3transformer_block/sequential/dense/Tensordot_1/axesConst*
_output_shapes
:*
dtype0*
valueB:�
3transformer_block/sequential/dense/Tensordot_1/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
4transformer_block/sequential/dense/Tensordot_1/ShapeShape;transformer_block/layer_normalization/batchnorm_1/add_1:z:0*
T0*
_output_shapes
:~
<transformer_block/sequential/dense/Tensordot_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot_1/GatherV2GatherV2=transformer_block/sequential/dense/Tensordot_1/Shape:output:0<transformer_block/sequential/dense/Tensordot_1/free:output:0Etransformer_block/sequential/dense/Tensordot_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
>transformer_block/sequential/dense/Tensordot_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense/Tensordot_1/GatherV2_1GatherV2=transformer_block/sequential/dense/Tensordot_1/Shape:output:0<transformer_block/sequential/dense/Tensordot_1/axes:output:0Gtransformer_block/sequential/dense/Tensordot_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4transformer_block/sequential/dense/Tensordot_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
3transformer_block/sequential/dense/Tensordot_1/ProdProd@transformer_block/sequential/dense/Tensordot_1/GatherV2:output:0=transformer_block/sequential/dense/Tensordot_1/Const:output:0*
T0*
_output_shapes
: �
6transformer_block/sequential/dense/Tensordot_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense/Tensordot_1/Prod_1ProdBtransformer_block/sequential/dense/Tensordot_1/GatherV2_1:output:0?transformer_block/sequential/dense/Tensordot_1/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_block/sequential/dense/Tensordot_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5transformer_block/sequential/dense/Tensordot_1/concatConcatV2<transformer_block/sequential/dense/Tensordot_1/free:output:0<transformer_block/sequential/dense/Tensordot_1/axes:output:0Ctransformer_block/sequential/dense/Tensordot_1/concat/axis:output:0*
N*
T0*
_output_shapes
:�
4transformer_block/sequential/dense/Tensordot_1/stackPack<transformer_block/sequential/dense/Tensordot_1/Prod:output:0>transformer_block/sequential/dense/Tensordot_1/Prod_1:output:0*
N*
T0*
_output_shapes
:�
8transformer_block/sequential/dense/Tensordot_1/transpose	Transpose;transformer_block/layer_normalization/batchnorm_1/add_1:z:0>transformer_block/sequential/dense/Tensordot_1/concat:output:0*
T0*,
_output_shapes
:���������-��
6transformer_block/sequential/dense/Tensordot_1/ReshapeReshape<transformer_block/sequential/dense/Tensordot_1/transpose:y:0=transformer_block/sequential/dense/Tensordot_1/stack:output:0*
T0*0
_output_shapes
:�������������������
5transformer_block/sequential/dense/Tensordot_1/MatMulMatMul?transformer_block/sequential/dense/Tensordot_1/Reshape:output:0Etransformer_block/sequential/dense/Tensordot_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
6transformer_block/sequential/dense/Tensordot_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�~
<transformer_block/sequential/dense/Tensordot_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot_1/concat_1ConcatV2@transformer_block/sequential/dense/Tensordot_1/GatherV2:output:0?transformer_block/sequential/dense/Tensordot_1/Const_2:output:0Etransformer_block/sequential/dense/Tensordot_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
.transformer_block/sequential/dense/Tensordot_1Reshape?transformer_block/sequential/dense/Tensordot_1/MatMul:product:0@transformer_block/sequential/dense/Tensordot_1/concat_1:output:0*
T0*,
_output_shapes
:���������-��
;transformer_block/sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOpBtransformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,transformer_block/sequential/dense/BiasAdd_1BiasAdd7transformer_block/sequential/dense/Tensordot_1:output:0Ctransformer_block/sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
)transformer_block/sequential/dense/Selu_1Selu5transformer_block/sequential/dense/BiasAdd_1:output:0*
T0*,
_output_shapes
:���������-��
?transformer_block/sequential/dense_1/Tensordot_1/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0
5transformer_block/sequential/dense_1/Tensordot_1/axesConst*
_output_shapes
:*
dtype0*
valueB:�
5transformer_block/sequential/dense_1/Tensordot_1/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
6transformer_block/sequential/dense_1/Tensordot_1/ShapeShape7transformer_block/sequential/dense/Selu_1:activations:0*
T0*
_output_shapes
:�
>transformer_block/sequential/dense_1/Tensordot_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot_1/GatherV2GatherV2?transformer_block/sequential/dense_1/Tensordot_1/Shape:output:0>transformer_block/sequential/dense_1/Tensordot_1/free:output:0Gtransformer_block/sequential/dense_1/Tensordot_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
@transformer_block/sequential/dense_1/Tensordot_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_block/sequential/dense_1/Tensordot_1/GatherV2_1GatherV2?transformer_block/sequential/dense_1/Tensordot_1/Shape:output:0>transformer_block/sequential/dense_1/Tensordot_1/axes:output:0Itransformer_block/sequential/dense_1/Tensordot_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
6transformer_block/sequential/dense_1/Tensordot_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense_1/Tensordot_1/ProdProdBtransformer_block/sequential/dense_1/Tensordot_1/GatherV2:output:0?transformer_block/sequential/dense_1/Tensordot_1/Const:output:0*
T0*
_output_shapes
: �
8transformer_block/sequential/dense_1/Tensordot_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
7transformer_block/sequential/dense_1/Tensordot_1/Prod_1ProdDtransformer_block/sequential/dense_1/Tensordot_1/GatherV2_1:output:0Atransformer_block/sequential/dense_1/Tensordot_1/Const_1:output:0*
T0*
_output_shapes
: ~
<transformer_block/sequential/dense_1/Tensordot_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense_1/Tensordot_1/concatConcatV2>transformer_block/sequential/dense_1/Tensordot_1/free:output:0>transformer_block/sequential/dense_1/Tensordot_1/axes:output:0Etransformer_block/sequential/dense_1/Tensordot_1/concat/axis:output:0*
N*
T0*
_output_shapes
:�
6transformer_block/sequential/dense_1/Tensordot_1/stackPack>transformer_block/sequential/dense_1/Tensordot_1/Prod:output:0@transformer_block/sequential/dense_1/Tensordot_1/Prod_1:output:0*
N*
T0*
_output_shapes
:�
:transformer_block/sequential/dense_1/Tensordot_1/transpose	Transpose7transformer_block/sequential/dense/Selu_1:activations:0@transformer_block/sequential/dense_1/Tensordot_1/concat:output:0*
T0*,
_output_shapes
:���������-��
8transformer_block/sequential/dense_1/Tensordot_1/ReshapeReshape>transformer_block/sequential/dense_1/Tensordot_1/transpose:y:0?transformer_block/sequential/dense_1/Tensordot_1/stack:output:0*
T0*0
_output_shapes
:�������������������
7transformer_block/sequential/dense_1/Tensordot_1/MatMulMatMulAtransformer_block/sequential/dense_1/Tensordot_1/Reshape:output:0Gtransformer_block/sequential/dense_1/Tensordot_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
8transformer_block/sequential/dense_1/Tensordot_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
>transformer_block/sequential/dense_1/Tensordot_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot_1/concat_1ConcatV2Btransformer_block/sequential/dense_1/Tensordot_1/GatherV2:output:0Atransformer_block/sequential/dense_1/Tensordot_1/Const_2:output:0Gtransformer_block/sequential/dense_1/Tensordot_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
0transformer_block/sequential/dense_1/Tensordot_1ReshapeAtransformer_block/sequential/dense_1/Tensordot_1/MatMul:product:0Btransformer_block/sequential/dense_1/Tensordot_1/concat_1:output:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.transformer_block/sequential/dense_1/BiasAdd_1BiasAdd9transformer_block/sequential/dense_1/Tensordot_1:output:0Etransformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
&transformer_block/dropout_1/Identity_1Identity7transformer_block/sequential/dense_1/BiasAdd_1:output:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_3AddV2;transformer_block/layer_normalization/batchnorm_1/add_1:z:0/transformer_block/dropout_1/Identity_1:output:0*
T0*,
_output_shapes
:���������-��
Htransformer_block/layer_normalization_1/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
6transformer_block/layer_normalization_1/moments_1/meanMeantransformer_block/add_3:z:0Qtransformer_block/layer_normalization_1/moments_1/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
>transformer_block/layer_normalization_1/moments_1/StopGradientStopGradient?transformer_block/layer_normalization_1/moments_1/mean:output:0*
T0*+
_output_shapes
:���������-�
Ctransformer_block/layer_normalization_1/moments_1/SquaredDifferenceSquaredDifferencetransformer_block/add_3:z:0Gtransformer_block/layer_normalization_1/moments_1/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Ltransformer_block/layer_normalization_1/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:transformer_block/layer_normalization_1/moments_1/varianceMeanGtransformer_block/layer_normalization_1/moments_1/SquaredDifference:z:0Utransformer_block/layer_normalization_1/moments_1/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(~
9transformer_block/layer_normalization_1/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
7transformer_block/layer_normalization_1/batchnorm_1/addAddV2Ctransformer_block/layer_normalization_1/moments_1/variance:output:0Btransformer_block/layer_normalization_1/batchnorm_1/add/y:output:0*
T0*+
_output_shapes
:���������-�
9transformer_block/layer_normalization_1/batchnorm_1/RsqrtRsqrt;transformer_block/layer_normalization_1/batchnorm_1/add:z:0*
T0*+
_output_shapes
:���������-�
Ftransformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_block/layer_normalization_1/batchnorm_1/mulMul=transformer_block/layer_normalization_1/batchnorm_1/Rsqrt:y:0Ntransformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_1/mul_1Multransformer_block/add_3:z:0;transformer_block/layer_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_1/mul_2Mul?transformer_block/layer_normalization_1/moments_1/mean:output:0;transformer_block/layer_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:���������-��
Btransformer_block/layer_normalization_1/batchnorm_1/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_block/layer_normalization_1/batchnorm_1/subSubJtransformer_block/layer_normalization_1/batchnorm_1/ReadVariableOp:value:0=transformer_block/layer_normalization_1/batchnorm_1/mul_2:z:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_1/add_1AddV2=transformer_block/layer_normalization_1/batchnorm_1/mul_1:z:0;transformer_block/layer_normalization_1/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:���������-��
Ktransformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_block/multi_head_attention/query/einsum_2/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_1/add_1:z:0Stransformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Atransformer_block/multi_head_attention/query/add_2/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_block/multi_head_attention/query/add_2AddV2Etransformer_block/multi_head_attention/query/einsum_2/Einsum:output:0Itransformer_block/multi_head_attention/query/add_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Itransformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOpReadVariableOpPtransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
:transformer_block/multi_head_attention/key/einsum_2/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_1/add_1:z:0Qtransformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
?transformer_block/multi_head_attention/key/add_2/ReadVariableOpReadVariableOpFtransformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
0transformer_block/multi_head_attention/key/add_2AddV2Ctransformer_block/multi_head_attention/key/einsum_2/Einsum:output:0Gtransformer_block/multi_head_attention/key/add_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Ktransformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_block/multi_head_attention/value/einsum_2/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_1/add_1:z:0Stransformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Atransformer_block/multi_head_attention/value/add_2/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_block/multi_head_attention/value/add_2AddV2Etransformer_block/multi_head_attention/value/einsum_2/Einsum:output:0Itransformer_block/multi_head_attention/value/add_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�s
.transformer_block/multi_head_attention/Mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
,transformer_block/multi_head_attention/Mul_2Mul6transformer_block/multi_head_attention/query/add_2:z:07transformer_block/multi_head_attention/Mul_2/y:output:0*
T0*0
_output_shapes
:���������-��
6transformer_block/multi_head_attention/einsum_4/EinsumEinsum4transformer_block/multi_head_attention/key/add_2:z:00transformer_block/multi_head_attention/Mul_2:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
8transformer_block/multi_head_attention/softmax/Softmax_2Softmax?transformer_block/multi_head_attention/einsum_4/Einsum:output:0*
T0*/
_output_shapes
:���������--�
9transformer_block/multi_head_attention/dropout/Identity_2IdentityBtransformer_block/multi_head_attention/softmax/Softmax_2:softmax:0*
T0*/
_output_shapes
:���������--�
6transformer_block/multi_head_attention/einsum_5/EinsumEinsumBtransformer_block/multi_head_attention/dropout/Identity_2:output:06transformer_block/multi_head_attention/value/add_2:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
Vtransformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOpReadVariableOp]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Gtransformer_block/multi_head_attention/attention_output/einsum_2/EinsumEinsum?transformer_block/multi_head_attention/einsum_5/Einsum:output:0^transformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
Ltransformer_block/multi_head_attention/attention_output/add_2/ReadVariableOpReadVariableOpStransformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=transformer_block/multi_head_attention/attention_output/add_2AddV2Ptransformer_block/multi_head_attention/attention_output/einsum_2/Einsum:output:0Ttransformer_block/multi_head_attention/attention_output/add_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
$transformer_block/dropout/Identity_2IdentityAtransformer_block/multi_head_attention/attention_output/add_2:z:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_4AddV2=transformer_block/layer_normalization_1/batchnorm_1/add_1:z:0-transformer_block/dropout/Identity_2:output:0*
T0*,
_output_shapes
:���������-��
Ftransformer_block/layer_normalization/moments_2/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
4transformer_block/layer_normalization/moments_2/meanMeantransformer_block/add_4:z:0Otransformer_block/layer_normalization/moments_2/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
<transformer_block/layer_normalization/moments_2/StopGradientStopGradient=transformer_block/layer_normalization/moments_2/mean:output:0*
T0*+
_output_shapes
:���������-�
Atransformer_block/layer_normalization/moments_2/SquaredDifferenceSquaredDifferencetransformer_block/add_4:z:0Etransformer_block/layer_normalization/moments_2/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Jtransformer_block/layer_normalization/moments_2/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
8transformer_block/layer_normalization/moments_2/varianceMeanEtransformer_block/layer_normalization/moments_2/SquaredDifference:z:0Stransformer_block/layer_normalization/moments_2/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(|
7transformer_block/layer_normalization/batchnorm_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
5transformer_block/layer_normalization/batchnorm_2/addAddV2Atransformer_block/layer_normalization/moments_2/variance:output:0@transformer_block/layer_normalization/batchnorm_2/add/y:output:0*
T0*+
_output_shapes
:���������-�
7transformer_block/layer_normalization/batchnorm_2/RsqrtRsqrt9transformer_block/layer_normalization/batchnorm_2/add:z:0*
T0*+
_output_shapes
:���������-�
Dtransformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization/batchnorm_2/mulMul;transformer_block/layer_normalization/batchnorm_2/Rsqrt:y:0Ltransformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_2/mul_1Multransformer_block/add_4:z:09transformer_block/layer_normalization/batchnorm_2/mul:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_2/mul_2Mul=transformer_block/layer_normalization/moments_2/mean:output:09transformer_block/layer_normalization/batchnorm_2/mul:z:0*
T0*,
_output_shapes
:���������-��
@transformer_block/layer_normalization/batchnorm_2/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization/batchnorm_2/subSubHtransformer_block/layer_normalization/batchnorm_2/ReadVariableOp:value:0;transformer_block/layer_normalization/batchnorm_2/mul_2:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_2/add_1AddV2;transformer_block/layer_normalization/batchnorm_2/mul_1:z:09transformer_block/layer_normalization/batchnorm_2/sub:z:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense/Tensordot_2/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0}
3transformer_block/sequential/dense/Tensordot_2/axesConst*
_output_shapes
:*
dtype0*
valueB:�
3transformer_block/sequential/dense/Tensordot_2/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
4transformer_block/sequential/dense/Tensordot_2/ShapeShape;transformer_block/layer_normalization/batchnorm_2/add_1:z:0*
T0*
_output_shapes
:~
<transformer_block/sequential/dense/Tensordot_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot_2/GatherV2GatherV2=transformer_block/sequential/dense/Tensordot_2/Shape:output:0<transformer_block/sequential/dense/Tensordot_2/free:output:0Etransformer_block/sequential/dense/Tensordot_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
>transformer_block/sequential/dense/Tensordot_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense/Tensordot_2/GatherV2_1GatherV2=transformer_block/sequential/dense/Tensordot_2/Shape:output:0<transformer_block/sequential/dense/Tensordot_2/axes:output:0Gtransformer_block/sequential/dense/Tensordot_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4transformer_block/sequential/dense/Tensordot_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
3transformer_block/sequential/dense/Tensordot_2/ProdProd@transformer_block/sequential/dense/Tensordot_2/GatherV2:output:0=transformer_block/sequential/dense/Tensordot_2/Const:output:0*
T0*
_output_shapes
: �
6transformer_block/sequential/dense/Tensordot_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense/Tensordot_2/Prod_1ProdBtransformer_block/sequential/dense/Tensordot_2/GatherV2_1:output:0?transformer_block/sequential/dense/Tensordot_2/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_block/sequential/dense/Tensordot_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5transformer_block/sequential/dense/Tensordot_2/concatConcatV2<transformer_block/sequential/dense/Tensordot_2/free:output:0<transformer_block/sequential/dense/Tensordot_2/axes:output:0Ctransformer_block/sequential/dense/Tensordot_2/concat/axis:output:0*
N*
T0*
_output_shapes
:�
4transformer_block/sequential/dense/Tensordot_2/stackPack<transformer_block/sequential/dense/Tensordot_2/Prod:output:0>transformer_block/sequential/dense/Tensordot_2/Prod_1:output:0*
N*
T0*
_output_shapes
:�
8transformer_block/sequential/dense/Tensordot_2/transpose	Transpose;transformer_block/layer_normalization/batchnorm_2/add_1:z:0>transformer_block/sequential/dense/Tensordot_2/concat:output:0*
T0*,
_output_shapes
:���������-��
6transformer_block/sequential/dense/Tensordot_2/ReshapeReshape<transformer_block/sequential/dense/Tensordot_2/transpose:y:0=transformer_block/sequential/dense/Tensordot_2/stack:output:0*
T0*0
_output_shapes
:�������������������
5transformer_block/sequential/dense/Tensordot_2/MatMulMatMul?transformer_block/sequential/dense/Tensordot_2/Reshape:output:0Etransformer_block/sequential/dense/Tensordot_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
6transformer_block/sequential/dense/Tensordot_2/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�~
<transformer_block/sequential/dense/Tensordot_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot_2/concat_1ConcatV2@transformer_block/sequential/dense/Tensordot_2/GatherV2:output:0?transformer_block/sequential/dense/Tensordot_2/Const_2:output:0Etransformer_block/sequential/dense/Tensordot_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
.transformer_block/sequential/dense/Tensordot_2Reshape?transformer_block/sequential/dense/Tensordot_2/MatMul:product:0@transformer_block/sequential/dense/Tensordot_2/concat_1:output:0*
T0*,
_output_shapes
:���������-��
;transformer_block/sequential/dense/BiasAdd_2/ReadVariableOpReadVariableOpBtransformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,transformer_block/sequential/dense/BiasAdd_2BiasAdd7transformer_block/sequential/dense/Tensordot_2:output:0Ctransformer_block/sequential/dense/BiasAdd_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
)transformer_block/sequential/dense/Selu_2Selu5transformer_block/sequential/dense/BiasAdd_2:output:0*
T0*,
_output_shapes
:���������-��
?transformer_block/sequential/dense_1/Tensordot_2/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0
5transformer_block/sequential/dense_1/Tensordot_2/axesConst*
_output_shapes
:*
dtype0*
valueB:�
5transformer_block/sequential/dense_1/Tensordot_2/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
6transformer_block/sequential/dense_1/Tensordot_2/ShapeShape7transformer_block/sequential/dense/Selu_2:activations:0*
T0*
_output_shapes
:�
>transformer_block/sequential/dense_1/Tensordot_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot_2/GatherV2GatherV2?transformer_block/sequential/dense_1/Tensordot_2/Shape:output:0>transformer_block/sequential/dense_1/Tensordot_2/free:output:0Gtransformer_block/sequential/dense_1/Tensordot_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
@transformer_block/sequential/dense_1/Tensordot_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_block/sequential/dense_1/Tensordot_2/GatherV2_1GatherV2?transformer_block/sequential/dense_1/Tensordot_2/Shape:output:0>transformer_block/sequential/dense_1/Tensordot_2/axes:output:0Itransformer_block/sequential/dense_1/Tensordot_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
6transformer_block/sequential/dense_1/Tensordot_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense_1/Tensordot_2/ProdProdBtransformer_block/sequential/dense_1/Tensordot_2/GatherV2:output:0?transformer_block/sequential/dense_1/Tensordot_2/Const:output:0*
T0*
_output_shapes
: �
8transformer_block/sequential/dense_1/Tensordot_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
7transformer_block/sequential/dense_1/Tensordot_2/Prod_1ProdDtransformer_block/sequential/dense_1/Tensordot_2/GatherV2_1:output:0Atransformer_block/sequential/dense_1/Tensordot_2/Const_1:output:0*
T0*
_output_shapes
: ~
<transformer_block/sequential/dense_1/Tensordot_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense_1/Tensordot_2/concatConcatV2>transformer_block/sequential/dense_1/Tensordot_2/free:output:0>transformer_block/sequential/dense_1/Tensordot_2/axes:output:0Etransformer_block/sequential/dense_1/Tensordot_2/concat/axis:output:0*
N*
T0*
_output_shapes
:�
6transformer_block/sequential/dense_1/Tensordot_2/stackPack>transformer_block/sequential/dense_1/Tensordot_2/Prod:output:0@transformer_block/sequential/dense_1/Tensordot_2/Prod_1:output:0*
N*
T0*
_output_shapes
:�
:transformer_block/sequential/dense_1/Tensordot_2/transpose	Transpose7transformer_block/sequential/dense/Selu_2:activations:0@transformer_block/sequential/dense_1/Tensordot_2/concat:output:0*
T0*,
_output_shapes
:���������-��
8transformer_block/sequential/dense_1/Tensordot_2/ReshapeReshape>transformer_block/sequential/dense_1/Tensordot_2/transpose:y:0?transformer_block/sequential/dense_1/Tensordot_2/stack:output:0*
T0*0
_output_shapes
:�������������������
7transformer_block/sequential/dense_1/Tensordot_2/MatMulMatMulAtransformer_block/sequential/dense_1/Tensordot_2/Reshape:output:0Gtransformer_block/sequential/dense_1/Tensordot_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
8transformer_block/sequential/dense_1/Tensordot_2/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
>transformer_block/sequential/dense_1/Tensordot_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot_2/concat_1ConcatV2Btransformer_block/sequential/dense_1/Tensordot_2/GatherV2:output:0Atransformer_block/sequential/dense_1/Tensordot_2/Const_2:output:0Gtransformer_block/sequential/dense_1/Tensordot_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
0transformer_block/sequential/dense_1/Tensordot_2ReshapeAtransformer_block/sequential/dense_1/Tensordot_2/MatMul:product:0Btransformer_block/sequential/dense_1/Tensordot_2/concat_1:output:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.transformer_block/sequential/dense_1/BiasAdd_2BiasAdd9transformer_block/sequential/dense_1/Tensordot_2:output:0Etransformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
&transformer_block/dropout_1/Identity_2Identity7transformer_block/sequential/dense_1/BiasAdd_2:output:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_5AddV2;transformer_block/layer_normalization/batchnorm_2/add_1:z:0/transformer_block/dropout_1/Identity_2:output:0*
T0*,
_output_shapes
:���������-��
Htransformer_block/layer_normalization_1/moments_2/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
6transformer_block/layer_normalization_1/moments_2/meanMeantransformer_block/add_5:z:0Qtransformer_block/layer_normalization_1/moments_2/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
>transformer_block/layer_normalization_1/moments_2/StopGradientStopGradient?transformer_block/layer_normalization_1/moments_2/mean:output:0*
T0*+
_output_shapes
:���������-�
Ctransformer_block/layer_normalization_1/moments_2/SquaredDifferenceSquaredDifferencetransformer_block/add_5:z:0Gtransformer_block/layer_normalization_1/moments_2/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Ltransformer_block/layer_normalization_1/moments_2/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:transformer_block/layer_normalization_1/moments_2/varianceMeanGtransformer_block/layer_normalization_1/moments_2/SquaredDifference:z:0Utransformer_block/layer_normalization_1/moments_2/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(~
9transformer_block/layer_normalization_1/batchnorm_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
7transformer_block/layer_normalization_1/batchnorm_2/addAddV2Ctransformer_block/layer_normalization_1/moments_2/variance:output:0Btransformer_block/layer_normalization_1/batchnorm_2/add/y:output:0*
T0*+
_output_shapes
:���������-�
9transformer_block/layer_normalization_1/batchnorm_2/RsqrtRsqrt;transformer_block/layer_normalization_1/batchnorm_2/add:z:0*
T0*+
_output_shapes
:���������-�
Ftransformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_block/layer_normalization_1/batchnorm_2/mulMul=transformer_block/layer_normalization_1/batchnorm_2/Rsqrt:y:0Ntransformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_2/mul_1Multransformer_block/add_5:z:0;transformer_block/layer_normalization_1/batchnorm_2/mul:z:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_2/mul_2Mul?transformer_block/layer_normalization_1/moments_2/mean:output:0;transformer_block/layer_normalization_1/batchnorm_2/mul:z:0*
T0*,
_output_shapes
:���������-��
Btransformer_block/layer_normalization_1/batchnorm_2/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_block/layer_normalization_1/batchnorm_2/subSubJtransformer_block/layer_normalization_1/batchnorm_2/ReadVariableOp:value:0=transformer_block/layer_normalization_1/batchnorm_2/mul_2:z:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_2/add_1AddV2=transformer_block/layer_normalization_1/batchnorm_2/mul_1:z:0;transformer_block/layer_normalization_1/batchnorm_2/sub:z:0*
T0*,
_output_shapes
:���������-��
Ktransformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_block/multi_head_attention/query/einsum_3/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_2/add_1:z:0Stransformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Atransformer_block/multi_head_attention/query/add_3/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_block/multi_head_attention/query/add_3AddV2Etransformer_block/multi_head_attention/query/einsum_3/Einsum:output:0Itransformer_block/multi_head_attention/query/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Itransformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOpReadVariableOpPtransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
:transformer_block/multi_head_attention/key/einsum_3/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_2/add_1:z:0Qtransformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
?transformer_block/multi_head_attention/key/add_3/ReadVariableOpReadVariableOpFtransformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
0transformer_block/multi_head_attention/key/add_3AddV2Ctransformer_block/multi_head_attention/key/einsum_3/Einsum:output:0Gtransformer_block/multi_head_attention/key/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Ktransformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_block/multi_head_attention/value/einsum_3/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_2/add_1:z:0Stransformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Atransformer_block/multi_head_attention/value/add_3/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_block/multi_head_attention/value/add_3AddV2Etransformer_block/multi_head_attention/value/einsum_3/Einsum:output:0Itransformer_block/multi_head_attention/value/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�s
.transformer_block/multi_head_attention/Mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
,transformer_block/multi_head_attention/Mul_3Mul6transformer_block/multi_head_attention/query/add_3:z:07transformer_block/multi_head_attention/Mul_3/y:output:0*
T0*0
_output_shapes
:���������-��
6transformer_block/multi_head_attention/einsum_6/EinsumEinsum4transformer_block/multi_head_attention/key/add_3:z:00transformer_block/multi_head_attention/Mul_3:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
8transformer_block/multi_head_attention/softmax/Softmax_3Softmax?transformer_block/multi_head_attention/einsum_6/Einsum:output:0*
T0*/
_output_shapes
:���������--�
9transformer_block/multi_head_attention/dropout/Identity_3IdentityBtransformer_block/multi_head_attention/softmax/Softmax_3:softmax:0*
T0*/
_output_shapes
:���������--�
6transformer_block/multi_head_attention/einsum_7/EinsumEinsumBtransformer_block/multi_head_attention/dropout/Identity_3:output:06transformer_block/multi_head_attention/value/add_3:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
Vtransformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOpReadVariableOp]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Gtransformer_block/multi_head_attention/attention_output/einsum_3/EinsumEinsum?transformer_block/multi_head_attention/einsum_7/Einsum:output:0^transformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
Ltransformer_block/multi_head_attention/attention_output/add_3/ReadVariableOpReadVariableOpStransformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=transformer_block/multi_head_attention/attention_output/add_3AddV2Ptransformer_block/multi_head_attention/attention_output/einsum_3/Einsum:output:0Ttransformer_block/multi_head_attention/attention_output/add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
$transformer_block/dropout/Identity_3IdentityAtransformer_block/multi_head_attention/attention_output/add_3:z:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_6AddV2=transformer_block/layer_normalization_1/batchnorm_2/add_1:z:0-transformer_block/dropout/Identity_3:output:0*
T0*,
_output_shapes
:���������-��
Ftransformer_block/layer_normalization/moments_3/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
4transformer_block/layer_normalization/moments_3/meanMeantransformer_block/add_6:z:0Otransformer_block/layer_normalization/moments_3/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
<transformer_block/layer_normalization/moments_3/StopGradientStopGradient=transformer_block/layer_normalization/moments_3/mean:output:0*
T0*+
_output_shapes
:���������-�
Atransformer_block/layer_normalization/moments_3/SquaredDifferenceSquaredDifferencetransformer_block/add_6:z:0Etransformer_block/layer_normalization/moments_3/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Jtransformer_block/layer_normalization/moments_3/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
8transformer_block/layer_normalization/moments_3/varianceMeanEtransformer_block/layer_normalization/moments_3/SquaredDifference:z:0Stransformer_block/layer_normalization/moments_3/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(|
7transformer_block/layer_normalization/batchnorm_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
5transformer_block/layer_normalization/batchnorm_3/addAddV2Atransformer_block/layer_normalization/moments_3/variance:output:0@transformer_block/layer_normalization/batchnorm_3/add/y:output:0*
T0*+
_output_shapes
:���������-�
7transformer_block/layer_normalization/batchnorm_3/RsqrtRsqrt9transformer_block/layer_normalization/batchnorm_3/add:z:0*
T0*+
_output_shapes
:���������-�
Dtransformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization/batchnorm_3/mulMul;transformer_block/layer_normalization/batchnorm_3/Rsqrt:y:0Ltransformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_3/mul_1Multransformer_block/add_6:z:09transformer_block/layer_normalization/batchnorm_3/mul:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_3/mul_2Mul=transformer_block/layer_normalization/moments_3/mean:output:09transformer_block/layer_normalization/batchnorm_3/mul:z:0*
T0*,
_output_shapes
:���������-��
@transformer_block/layer_normalization/batchnorm_3/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization/batchnorm_3/subSubHtransformer_block/layer_normalization/batchnorm_3/ReadVariableOp:value:0;transformer_block/layer_normalization/batchnorm_3/mul_2:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_3/add_1AddV2;transformer_block/layer_normalization/batchnorm_3/mul_1:z:09transformer_block/layer_normalization/batchnorm_3/sub:z:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense/Tensordot_3/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0}
3transformer_block/sequential/dense/Tensordot_3/axesConst*
_output_shapes
:*
dtype0*
valueB:�
3transformer_block/sequential/dense/Tensordot_3/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
4transformer_block/sequential/dense/Tensordot_3/ShapeShape;transformer_block/layer_normalization/batchnorm_3/add_1:z:0*
T0*
_output_shapes
:~
<transformer_block/sequential/dense/Tensordot_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot_3/GatherV2GatherV2=transformer_block/sequential/dense/Tensordot_3/Shape:output:0<transformer_block/sequential/dense/Tensordot_3/free:output:0Etransformer_block/sequential/dense/Tensordot_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
>transformer_block/sequential/dense/Tensordot_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense/Tensordot_3/GatherV2_1GatherV2=transformer_block/sequential/dense/Tensordot_3/Shape:output:0<transformer_block/sequential/dense/Tensordot_3/axes:output:0Gtransformer_block/sequential/dense/Tensordot_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4transformer_block/sequential/dense/Tensordot_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
3transformer_block/sequential/dense/Tensordot_3/ProdProd@transformer_block/sequential/dense/Tensordot_3/GatherV2:output:0=transformer_block/sequential/dense/Tensordot_3/Const:output:0*
T0*
_output_shapes
: �
6transformer_block/sequential/dense/Tensordot_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense/Tensordot_3/Prod_1ProdBtransformer_block/sequential/dense/Tensordot_3/GatherV2_1:output:0?transformer_block/sequential/dense/Tensordot_3/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_block/sequential/dense/Tensordot_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5transformer_block/sequential/dense/Tensordot_3/concatConcatV2<transformer_block/sequential/dense/Tensordot_3/free:output:0<transformer_block/sequential/dense/Tensordot_3/axes:output:0Ctransformer_block/sequential/dense/Tensordot_3/concat/axis:output:0*
N*
T0*
_output_shapes
:�
4transformer_block/sequential/dense/Tensordot_3/stackPack<transformer_block/sequential/dense/Tensordot_3/Prod:output:0>transformer_block/sequential/dense/Tensordot_3/Prod_1:output:0*
N*
T0*
_output_shapes
:�
8transformer_block/sequential/dense/Tensordot_3/transpose	Transpose;transformer_block/layer_normalization/batchnorm_3/add_1:z:0>transformer_block/sequential/dense/Tensordot_3/concat:output:0*
T0*,
_output_shapes
:���������-��
6transformer_block/sequential/dense/Tensordot_3/ReshapeReshape<transformer_block/sequential/dense/Tensordot_3/transpose:y:0=transformer_block/sequential/dense/Tensordot_3/stack:output:0*
T0*0
_output_shapes
:�������������������
5transformer_block/sequential/dense/Tensordot_3/MatMulMatMul?transformer_block/sequential/dense/Tensordot_3/Reshape:output:0Etransformer_block/sequential/dense/Tensordot_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
6transformer_block/sequential/dense/Tensordot_3/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�~
<transformer_block/sequential/dense/Tensordot_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot_3/concat_1ConcatV2@transformer_block/sequential/dense/Tensordot_3/GatherV2:output:0?transformer_block/sequential/dense/Tensordot_3/Const_2:output:0Etransformer_block/sequential/dense/Tensordot_3/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
.transformer_block/sequential/dense/Tensordot_3Reshape?transformer_block/sequential/dense/Tensordot_3/MatMul:product:0@transformer_block/sequential/dense/Tensordot_3/concat_1:output:0*
T0*,
_output_shapes
:���������-��
;transformer_block/sequential/dense/BiasAdd_3/ReadVariableOpReadVariableOpBtransformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,transformer_block/sequential/dense/BiasAdd_3BiasAdd7transformer_block/sequential/dense/Tensordot_3:output:0Ctransformer_block/sequential/dense/BiasAdd_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
)transformer_block/sequential/dense/Selu_3Selu5transformer_block/sequential/dense/BiasAdd_3:output:0*
T0*,
_output_shapes
:���������-��
?transformer_block/sequential/dense_1/Tensordot_3/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0
5transformer_block/sequential/dense_1/Tensordot_3/axesConst*
_output_shapes
:*
dtype0*
valueB:�
5transformer_block/sequential/dense_1/Tensordot_3/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
6transformer_block/sequential/dense_1/Tensordot_3/ShapeShape7transformer_block/sequential/dense/Selu_3:activations:0*
T0*
_output_shapes
:�
>transformer_block/sequential/dense_1/Tensordot_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot_3/GatherV2GatherV2?transformer_block/sequential/dense_1/Tensordot_3/Shape:output:0>transformer_block/sequential/dense_1/Tensordot_3/free:output:0Gtransformer_block/sequential/dense_1/Tensordot_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
@transformer_block/sequential/dense_1/Tensordot_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_block/sequential/dense_1/Tensordot_3/GatherV2_1GatherV2?transformer_block/sequential/dense_1/Tensordot_3/Shape:output:0>transformer_block/sequential/dense_1/Tensordot_3/axes:output:0Itransformer_block/sequential/dense_1/Tensordot_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
6transformer_block/sequential/dense_1/Tensordot_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense_1/Tensordot_3/ProdProdBtransformer_block/sequential/dense_1/Tensordot_3/GatherV2:output:0?transformer_block/sequential/dense_1/Tensordot_3/Const:output:0*
T0*
_output_shapes
: �
8transformer_block/sequential/dense_1/Tensordot_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
7transformer_block/sequential/dense_1/Tensordot_3/Prod_1ProdDtransformer_block/sequential/dense_1/Tensordot_3/GatherV2_1:output:0Atransformer_block/sequential/dense_1/Tensordot_3/Const_1:output:0*
T0*
_output_shapes
: ~
<transformer_block/sequential/dense_1/Tensordot_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense_1/Tensordot_3/concatConcatV2>transformer_block/sequential/dense_1/Tensordot_3/free:output:0>transformer_block/sequential/dense_1/Tensordot_3/axes:output:0Etransformer_block/sequential/dense_1/Tensordot_3/concat/axis:output:0*
N*
T0*
_output_shapes
:�
6transformer_block/sequential/dense_1/Tensordot_3/stackPack>transformer_block/sequential/dense_1/Tensordot_3/Prod:output:0@transformer_block/sequential/dense_1/Tensordot_3/Prod_1:output:0*
N*
T0*
_output_shapes
:�
:transformer_block/sequential/dense_1/Tensordot_3/transpose	Transpose7transformer_block/sequential/dense/Selu_3:activations:0@transformer_block/sequential/dense_1/Tensordot_3/concat:output:0*
T0*,
_output_shapes
:���������-��
8transformer_block/sequential/dense_1/Tensordot_3/ReshapeReshape>transformer_block/sequential/dense_1/Tensordot_3/transpose:y:0?transformer_block/sequential/dense_1/Tensordot_3/stack:output:0*
T0*0
_output_shapes
:�������������������
7transformer_block/sequential/dense_1/Tensordot_3/MatMulMatMulAtransformer_block/sequential/dense_1/Tensordot_3/Reshape:output:0Gtransformer_block/sequential/dense_1/Tensordot_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
8transformer_block/sequential/dense_1/Tensordot_3/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
>transformer_block/sequential/dense_1/Tensordot_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot_3/concat_1ConcatV2Btransformer_block/sequential/dense_1/Tensordot_3/GatherV2:output:0Atransformer_block/sequential/dense_1/Tensordot_3/Const_2:output:0Gtransformer_block/sequential/dense_1/Tensordot_3/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
0transformer_block/sequential/dense_1/Tensordot_3ReshapeAtransformer_block/sequential/dense_1/Tensordot_3/MatMul:product:0Btransformer_block/sequential/dense_1/Tensordot_3/concat_1:output:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.transformer_block/sequential/dense_1/BiasAdd_3BiasAdd9transformer_block/sequential/dense_1/Tensordot_3:output:0Etransformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
&transformer_block/dropout_1/Identity_3Identity7transformer_block/sequential/dense_1/BiasAdd_3:output:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_7AddV2;transformer_block/layer_normalization/batchnorm_3/add_1:z:0/transformer_block/dropout_1/Identity_3:output:0*
T0*,
_output_shapes
:���������-��
Htransformer_block/layer_normalization_1/moments_3/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
6transformer_block/layer_normalization_1/moments_3/meanMeantransformer_block/add_7:z:0Qtransformer_block/layer_normalization_1/moments_3/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
>transformer_block/layer_normalization_1/moments_3/StopGradientStopGradient?transformer_block/layer_normalization_1/moments_3/mean:output:0*
T0*+
_output_shapes
:���������-�
Ctransformer_block/layer_normalization_1/moments_3/SquaredDifferenceSquaredDifferencetransformer_block/add_7:z:0Gtransformer_block/layer_normalization_1/moments_3/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Ltransformer_block/layer_normalization_1/moments_3/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:transformer_block/layer_normalization_1/moments_3/varianceMeanGtransformer_block/layer_normalization_1/moments_3/SquaredDifference:z:0Utransformer_block/layer_normalization_1/moments_3/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(~
9transformer_block/layer_normalization_1/batchnorm_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
7transformer_block/layer_normalization_1/batchnorm_3/addAddV2Ctransformer_block/layer_normalization_1/moments_3/variance:output:0Btransformer_block/layer_normalization_1/batchnorm_3/add/y:output:0*
T0*+
_output_shapes
:���������-�
9transformer_block/layer_normalization_1/batchnorm_3/RsqrtRsqrt;transformer_block/layer_normalization_1/batchnorm_3/add:z:0*
T0*+
_output_shapes
:���������-�
Ftransformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_block/layer_normalization_1/batchnorm_3/mulMul=transformer_block/layer_normalization_1/batchnorm_3/Rsqrt:y:0Ntransformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_3/mul_1Multransformer_block/add_7:z:0;transformer_block/layer_normalization_1/batchnorm_3/mul:z:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_3/mul_2Mul?transformer_block/layer_normalization_1/moments_3/mean:output:0;transformer_block/layer_normalization_1/batchnorm_3/mul:z:0*
T0*,
_output_shapes
:���������-��
Btransformer_block/layer_normalization_1/batchnorm_3/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_block/layer_normalization_1/batchnorm_3/subSubJtransformer_block/layer_normalization_1/batchnorm_3/ReadVariableOp:value:0=transformer_block/layer_normalization_1/batchnorm_3/mul_2:z:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_3/add_1AddV2=transformer_block/layer_normalization_1/batchnorm_3/mul_1:z:0;transformer_block/layer_normalization_1/batchnorm_3/sub:z:0*
T0*,
_output_shapes
:���������-��
Ktransformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_block/multi_head_attention/query/einsum_4/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_3/add_1:z:0Stransformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Atransformer_block/multi_head_attention/query/add_4/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_block/multi_head_attention/query/add_4AddV2Etransformer_block/multi_head_attention/query/einsum_4/Einsum:output:0Itransformer_block/multi_head_attention/query/add_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Itransformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOpReadVariableOpPtransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
:transformer_block/multi_head_attention/key/einsum_4/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_3/add_1:z:0Qtransformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
?transformer_block/multi_head_attention/key/add_4/ReadVariableOpReadVariableOpFtransformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
0transformer_block/multi_head_attention/key/add_4AddV2Ctransformer_block/multi_head_attention/key/einsum_4/Einsum:output:0Gtransformer_block/multi_head_attention/key/add_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Ktransformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_block/multi_head_attention/value/einsum_4/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_3/add_1:z:0Stransformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Atransformer_block/multi_head_attention/value/add_4/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_block/multi_head_attention/value/add_4AddV2Etransformer_block/multi_head_attention/value/einsum_4/Einsum:output:0Itransformer_block/multi_head_attention/value/add_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�s
.transformer_block/multi_head_attention/Mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
,transformer_block/multi_head_attention/Mul_4Mul6transformer_block/multi_head_attention/query/add_4:z:07transformer_block/multi_head_attention/Mul_4/y:output:0*
T0*0
_output_shapes
:���������-��
6transformer_block/multi_head_attention/einsum_8/EinsumEinsum4transformer_block/multi_head_attention/key/add_4:z:00transformer_block/multi_head_attention/Mul_4:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
8transformer_block/multi_head_attention/softmax/Softmax_4Softmax?transformer_block/multi_head_attention/einsum_8/Einsum:output:0*
T0*/
_output_shapes
:���������--�
9transformer_block/multi_head_attention/dropout/Identity_4IdentityBtransformer_block/multi_head_attention/softmax/Softmax_4:softmax:0*
T0*/
_output_shapes
:���������--�
6transformer_block/multi_head_attention/einsum_9/EinsumEinsumBtransformer_block/multi_head_attention/dropout/Identity_4:output:06transformer_block/multi_head_attention/value/add_4:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
Vtransformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOpReadVariableOp]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Gtransformer_block/multi_head_attention/attention_output/einsum_4/EinsumEinsum?transformer_block/multi_head_attention/einsum_9/Einsum:output:0^transformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
Ltransformer_block/multi_head_attention/attention_output/add_4/ReadVariableOpReadVariableOpStransformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=transformer_block/multi_head_attention/attention_output/add_4AddV2Ptransformer_block/multi_head_attention/attention_output/einsum_4/Einsum:output:0Ttransformer_block/multi_head_attention/attention_output/add_4/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
$transformer_block/dropout/Identity_4IdentityAtransformer_block/multi_head_attention/attention_output/add_4:z:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_8AddV2=transformer_block/layer_normalization_1/batchnorm_3/add_1:z:0-transformer_block/dropout/Identity_4:output:0*
T0*,
_output_shapes
:���������-��
Ftransformer_block/layer_normalization/moments_4/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
4transformer_block/layer_normalization/moments_4/meanMeantransformer_block/add_8:z:0Otransformer_block/layer_normalization/moments_4/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
<transformer_block/layer_normalization/moments_4/StopGradientStopGradient=transformer_block/layer_normalization/moments_4/mean:output:0*
T0*+
_output_shapes
:���������-�
Atransformer_block/layer_normalization/moments_4/SquaredDifferenceSquaredDifferencetransformer_block/add_8:z:0Etransformer_block/layer_normalization/moments_4/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Jtransformer_block/layer_normalization/moments_4/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
8transformer_block/layer_normalization/moments_4/varianceMeanEtransformer_block/layer_normalization/moments_4/SquaredDifference:z:0Stransformer_block/layer_normalization/moments_4/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(|
7transformer_block/layer_normalization/batchnorm_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
5transformer_block/layer_normalization/batchnorm_4/addAddV2Atransformer_block/layer_normalization/moments_4/variance:output:0@transformer_block/layer_normalization/batchnorm_4/add/y:output:0*
T0*+
_output_shapes
:���������-�
7transformer_block/layer_normalization/batchnorm_4/RsqrtRsqrt9transformer_block/layer_normalization/batchnorm_4/add:z:0*
T0*+
_output_shapes
:���������-�
Dtransformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization/batchnorm_4/mulMul;transformer_block/layer_normalization/batchnorm_4/Rsqrt:y:0Ltransformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_4/mul_1Multransformer_block/add_8:z:09transformer_block/layer_normalization/batchnorm_4/mul:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_4/mul_2Mul=transformer_block/layer_normalization/moments_4/mean:output:09transformer_block/layer_normalization/batchnorm_4/mul:z:0*
T0*,
_output_shapes
:���������-��
@transformer_block/layer_normalization/batchnorm_4/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization/batchnorm_4/subSubHtransformer_block/layer_normalization/batchnorm_4/ReadVariableOp:value:0;transformer_block/layer_normalization/batchnorm_4/mul_2:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_4/add_1AddV2;transformer_block/layer_normalization/batchnorm_4/mul_1:z:09transformer_block/layer_normalization/batchnorm_4/sub:z:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense/Tensordot_4/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0}
3transformer_block/sequential/dense/Tensordot_4/axesConst*
_output_shapes
:*
dtype0*
valueB:�
3transformer_block/sequential/dense/Tensordot_4/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
4transformer_block/sequential/dense/Tensordot_4/ShapeShape;transformer_block/layer_normalization/batchnorm_4/add_1:z:0*
T0*
_output_shapes
:~
<transformer_block/sequential/dense/Tensordot_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot_4/GatherV2GatherV2=transformer_block/sequential/dense/Tensordot_4/Shape:output:0<transformer_block/sequential/dense/Tensordot_4/free:output:0Etransformer_block/sequential/dense/Tensordot_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
>transformer_block/sequential/dense/Tensordot_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense/Tensordot_4/GatherV2_1GatherV2=transformer_block/sequential/dense/Tensordot_4/Shape:output:0<transformer_block/sequential/dense/Tensordot_4/axes:output:0Gtransformer_block/sequential/dense/Tensordot_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4transformer_block/sequential/dense/Tensordot_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
3transformer_block/sequential/dense/Tensordot_4/ProdProd@transformer_block/sequential/dense/Tensordot_4/GatherV2:output:0=transformer_block/sequential/dense/Tensordot_4/Const:output:0*
T0*
_output_shapes
: �
6transformer_block/sequential/dense/Tensordot_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense/Tensordot_4/Prod_1ProdBtransformer_block/sequential/dense/Tensordot_4/GatherV2_1:output:0?transformer_block/sequential/dense/Tensordot_4/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_block/sequential/dense/Tensordot_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5transformer_block/sequential/dense/Tensordot_4/concatConcatV2<transformer_block/sequential/dense/Tensordot_4/free:output:0<transformer_block/sequential/dense/Tensordot_4/axes:output:0Ctransformer_block/sequential/dense/Tensordot_4/concat/axis:output:0*
N*
T0*
_output_shapes
:�
4transformer_block/sequential/dense/Tensordot_4/stackPack<transformer_block/sequential/dense/Tensordot_4/Prod:output:0>transformer_block/sequential/dense/Tensordot_4/Prod_1:output:0*
N*
T0*
_output_shapes
:�
8transformer_block/sequential/dense/Tensordot_4/transpose	Transpose;transformer_block/layer_normalization/batchnorm_4/add_1:z:0>transformer_block/sequential/dense/Tensordot_4/concat:output:0*
T0*,
_output_shapes
:���������-��
6transformer_block/sequential/dense/Tensordot_4/ReshapeReshape<transformer_block/sequential/dense/Tensordot_4/transpose:y:0=transformer_block/sequential/dense/Tensordot_4/stack:output:0*
T0*0
_output_shapes
:�������������������
5transformer_block/sequential/dense/Tensordot_4/MatMulMatMul?transformer_block/sequential/dense/Tensordot_4/Reshape:output:0Etransformer_block/sequential/dense/Tensordot_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
6transformer_block/sequential/dense/Tensordot_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�~
<transformer_block/sequential/dense/Tensordot_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot_4/concat_1ConcatV2@transformer_block/sequential/dense/Tensordot_4/GatherV2:output:0?transformer_block/sequential/dense/Tensordot_4/Const_2:output:0Etransformer_block/sequential/dense/Tensordot_4/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
.transformer_block/sequential/dense/Tensordot_4Reshape?transformer_block/sequential/dense/Tensordot_4/MatMul:product:0@transformer_block/sequential/dense/Tensordot_4/concat_1:output:0*
T0*,
_output_shapes
:���������-��
;transformer_block/sequential/dense/BiasAdd_4/ReadVariableOpReadVariableOpBtransformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,transformer_block/sequential/dense/BiasAdd_4BiasAdd7transformer_block/sequential/dense/Tensordot_4:output:0Ctransformer_block/sequential/dense/BiasAdd_4/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
)transformer_block/sequential/dense/Selu_4Selu5transformer_block/sequential/dense/BiasAdd_4:output:0*
T0*,
_output_shapes
:���������-��
?transformer_block/sequential/dense_1/Tensordot_4/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0
5transformer_block/sequential/dense_1/Tensordot_4/axesConst*
_output_shapes
:*
dtype0*
valueB:�
5transformer_block/sequential/dense_1/Tensordot_4/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
6transformer_block/sequential/dense_1/Tensordot_4/ShapeShape7transformer_block/sequential/dense/Selu_4:activations:0*
T0*
_output_shapes
:�
>transformer_block/sequential/dense_1/Tensordot_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot_4/GatherV2GatherV2?transformer_block/sequential/dense_1/Tensordot_4/Shape:output:0>transformer_block/sequential/dense_1/Tensordot_4/free:output:0Gtransformer_block/sequential/dense_1/Tensordot_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
@transformer_block/sequential/dense_1/Tensordot_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_block/sequential/dense_1/Tensordot_4/GatherV2_1GatherV2?transformer_block/sequential/dense_1/Tensordot_4/Shape:output:0>transformer_block/sequential/dense_1/Tensordot_4/axes:output:0Itransformer_block/sequential/dense_1/Tensordot_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
6transformer_block/sequential/dense_1/Tensordot_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense_1/Tensordot_4/ProdProdBtransformer_block/sequential/dense_1/Tensordot_4/GatherV2:output:0?transformer_block/sequential/dense_1/Tensordot_4/Const:output:0*
T0*
_output_shapes
: �
8transformer_block/sequential/dense_1/Tensordot_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
7transformer_block/sequential/dense_1/Tensordot_4/Prod_1ProdDtransformer_block/sequential/dense_1/Tensordot_4/GatherV2_1:output:0Atransformer_block/sequential/dense_1/Tensordot_4/Const_1:output:0*
T0*
_output_shapes
: ~
<transformer_block/sequential/dense_1/Tensordot_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense_1/Tensordot_4/concatConcatV2>transformer_block/sequential/dense_1/Tensordot_4/free:output:0>transformer_block/sequential/dense_1/Tensordot_4/axes:output:0Etransformer_block/sequential/dense_1/Tensordot_4/concat/axis:output:0*
N*
T0*
_output_shapes
:�
6transformer_block/sequential/dense_1/Tensordot_4/stackPack>transformer_block/sequential/dense_1/Tensordot_4/Prod:output:0@transformer_block/sequential/dense_1/Tensordot_4/Prod_1:output:0*
N*
T0*
_output_shapes
:�
:transformer_block/sequential/dense_1/Tensordot_4/transpose	Transpose7transformer_block/sequential/dense/Selu_4:activations:0@transformer_block/sequential/dense_1/Tensordot_4/concat:output:0*
T0*,
_output_shapes
:���������-��
8transformer_block/sequential/dense_1/Tensordot_4/ReshapeReshape>transformer_block/sequential/dense_1/Tensordot_4/transpose:y:0?transformer_block/sequential/dense_1/Tensordot_4/stack:output:0*
T0*0
_output_shapes
:�������������������
7transformer_block/sequential/dense_1/Tensordot_4/MatMulMatMulAtransformer_block/sequential/dense_1/Tensordot_4/Reshape:output:0Gtransformer_block/sequential/dense_1/Tensordot_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
8transformer_block/sequential/dense_1/Tensordot_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
>transformer_block/sequential/dense_1/Tensordot_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot_4/concat_1ConcatV2Btransformer_block/sequential/dense_1/Tensordot_4/GatherV2:output:0Atransformer_block/sequential/dense_1/Tensordot_4/Const_2:output:0Gtransformer_block/sequential/dense_1/Tensordot_4/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
0transformer_block/sequential/dense_1/Tensordot_4ReshapeAtransformer_block/sequential/dense_1/Tensordot_4/MatMul:product:0Btransformer_block/sequential/dense_1/Tensordot_4/concat_1:output:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.transformer_block/sequential/dense_1/BiasAdd_4BiasAdd9transformer_block/sequential/dense_1/Tensordot_4:output:0Etransformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
&transformer_block/dropout_1/Identity_4Identity7transformer_block/sequential/dense_1/BiasAdd_4:output:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_9AddV2;transformer_block/layer_normalization/batchnorm_4/add_1:z:0/transformer_block/dropout_1/Identity_4:output:0*
T0*,
_output_shapes
:���������-��
Htransformer_block/layer_normalization_1/moments_4/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
6transformer_block/layer_normalization_1/moments_4/meanMeantransformer_block/add_9:z:0Qtransformer_block/layer_normalization_1/moments_4/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
>transformer_block/layer_normalization_1/moments_4/StopGradientStopGradient?transformer_block/layer_normalization_1/moments_4/mean:output:0*
T0*+
_output_shapes
:���������-�
Ctransformer_block/layer_normalization_1/moments_4/SquaredDifferenceSquaredDifferencetransformer_block/add_9:z:0Gtransformer_block/layer_normalization_1/moments_4/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Ltransformer_block/layer_normalization_1/moments_4/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:transformer_block/layer_normalization_1/moments_4/varianceMeanGtransformer_block/layer_normalization_1/moments_4/SquaredDifference:z:0Utransformer_block/layer_normalization_1/moments_4/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(~
9transformer_block/layer_normalization_1/batchnorm_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
7transformer_block/layer_normalization_1/batchnorm_4/addAddV2Ctransformer_block/layer_normalization_1/moments_4/variance:output:0Btransformer_block/layer_normalization_1/batchnorm_4/add/y:output:0*
T0*+
_output_shapes
:���������-�
9transformer_block/layer_normalization_1/batchnorm_4/RsqrtRsqrt;transformer_block/layer_normalization_1/batchnorm_4/add:z:0*
T0*+
_output_shapes
:���������-�
Ftransformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_block/layer_normalization_1/batchnorm_4/mulMul=transformer_block/layer_normalization_1/batchnorm_4/Rsqrt:y:0Ntransformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_4/mul_1Multransformer_block/add_9:z:0;transformer_block/layer_normalization_1/batchnorm_4/mul:z:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_4/mul_2Mul?transformer_block/layer_normalization_1/moments_4/mean:output:0;transformer_block/layer_normalization_1/batchnorm_4/mul:z:0*
T0*,
_output_shapes
:���������-��
Btransformer_block/layer_normalization_1/batchnorm_4/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_block/layer_normalization_1/batchnorm_4/subSubJtransformer_block/layer_normalization_1/batchnorm_4/ReadVariableOp:value:0=transformer_block/layer_normalization_1/batchnorm_4/mul_2:z:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_4/add_1AddV2=transformer_block/layer_normalization_1/batchnorm_4/mul_1:z:0;transformer_block/layer_normalization_1/batchnorm_4/sub:z:0*
T0*,
_output_shapes
:���������-�z
reshape/ShapeShape=transformer_block/layer_normalization_1/batchnorm_4/add_1:z:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Z
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :�-�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
reshape/ReshapeReshape=transformer_block/layer_normalization_1/batchnorm_4/add_1:z:0reshape/Reshape/shape:output:0*
T0*,
_output_shapes
:����������-o
dropout_2/IdentityIdentityreshape/Reshape:output:0*
T0*,
_output_shapes
:����������-�
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
�-�*
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_2/Tensordot/ShapeShapedropout_2/Identity:output:0*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_2/Tensordot/transpose	Transposedropout_2/Identity:output:0!dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������-�
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������e
dense_2/SeluSeludense_2/BiasAdd:output:0*
T0*,
_output_shapes
:����������q
dropout_3/IdentityIdentitydense_2/Selu:activations:0*
T0*,
_output_shapes
:�����������
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0`
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_3/Tensordot/ShapeShapedropout_3/Identity:output:0*
T0*
_output_shapes
:a
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_3/Tensordot/transpose	Transposedropout_3/Identity:output:0!dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������e
dense_3/SeluSeludense_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������q
dropout_4/IdentityIdentitydense_3/Selu:activations:0*
T0*,
_output_shapes
:�����������
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_4/Tensordot/ShapeShapedropout_4/Identity:output:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_4/Tensordot/transpose	Transposedropout_4/Identity:output:0!dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������@�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@d
dense_4/SeluSeludense_4/BiasAdd:output:0*
T0*+
_output_shapes
:���������@p
dropout_5/IdentityIdentitydense_4/Selu:activations:0*
T0*+
_output_shapes
:���������@�
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0`
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_5/Tensordot/ShapeShapedropout_5/Identity:output:0*
T0*
_output_shapes
:a
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_5/Tensordot/transpose	Transposedropout_5/Identity:output:0!dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������@�
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������c
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:����������
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������d
dense_5/SeluSeludense_5/BiasAdd:output:0*
T0*+
_output_shapes
:���������p
dropout_6/IdentityIdentitydense_5/Selu:activations:0*
T0*+
_output_shapes
:����������
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_6/Tensordot/ShapeShapedropout_6/Identity:output:0*
T0*
_output_shapes
:a
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_6/Tensordot/transpose	Transposedropout_6/Identity:output:0!dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:����������
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������c
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:����������
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������d
dense_6/SeluSeludense_6/BiasAdd:output:0*
T0*+
_output_shapes
:���������m
IdentityIdentitydense_6/Selu:activations:0^NoOp*
T0*+
_output_shapes
:����������0
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/Tensordot/ReadVariableOp8^token_and_position_embedding/embedding/embedding_lookup:^token_and_position_embedding/embedding_1/embedding_lookup?^transformer_block/layer_normalization/batchnorm/ReadVariableOpC^transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpA^transformer_block/layer_normalization/batchnorm_1/ReadVariableOpE^transformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOpA^transformer_block/layer_normalization/batchnorm_2/ReadVariableOpE^transformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOpA^transformer_block/layer_normalization/batchnorm_3/ReadVariableOpE^transformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOpA^transformer_block/layer_normalization/batchnorm_4/ReadVariableOpE^transformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOpA^transformer_block/layer_normalization_1/batchnorm/ReadVariableOpE^transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpC^transformer_block/layer_normalization_1/batchnorm_1/ReadVariableOpG^transformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOpC^transformer_block/layer_normalization_1/batchnorm_2/ReadVariableOpG^transformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOpC^transformer_block/layer_normalization_1/batchnorm_3/ReadVariableOpG^transformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOpC^transformer_block/layer_normalization_1/batchnorm_4/ReadVariableOpG^transformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOpK^transformer_block/multi_head_attention/attention_output/add/ReadVariableOpM^transformer_block/multi_head_attention/attention_output/add_1/ReadVariableOpM^transformer_block/multi_head_attention/attention_output/add_2/ReadVariableOpM^transformer_block/multi_head_attention/attention_output/add_3/ReadVariableOpM^transformer_block/multi_head_attention/attention_output/add_4/ReadVariableOpU^transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpW^transformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOpW^transformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOpW^transformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOpW^transformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOp>^transformer_block/multi_head_attention/key/add/ReadVariableOp@^transformer_block/multi_head_attention/key/add_1/ReadVariableOp@^transformer_block/multi_head_attention/key/add_2/ReadVariableOp@^transformer_block/multi_head_attention/key/add_3/ReadVariableOp@^transformer_block/multi_head_attention/key/add_4/ReadVariableOpH^transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpJ^transformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOpJ^transformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOpJ^transformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOpJ^transformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOp@^transformer_block/multi_head_attention/query/add/ReadVariableOpB^transformer_block/multi_head_attention/query/add_1/ReadVariableOpB^transformer_block/multi_head_attention/query/add_2/ReadVariableOpB^transformer_block/multi_head_attention/query/add_3/ReadVariableOpB^transformer_block/multi_head_attention/query/add_4/ReadVariableOpJ^transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpL^transformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOpL^transformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOpL^transformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOpL^transformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOp@^transformer_block/multi_head_attention/value/add/ReadVariableOpB^transformer_block/multi_head_attention/value/add_1/ReadVariableOpB^transformer_block/multi_head_attention/value/add_2/ReadVariableOpB^transformer_block/multi_head_attention/value/add_3/ReadVariableOpB^transformer_block/multi_head_attention/value/add_4/ReadVariableOpJ^transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpL^transformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOpL^transformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOpL^transformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOpL^transformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOp:^transformer_block/sequential/dense/BiasAdd/ReadVariableOp<^transformer_block/sequential/dense/BiasAdd_1/ReadVariableOp<^transformer_block/sequential/dense/BiasAdd_2/ReadVariableOp<^transformer_block/sequential/dense/BiasAdd_3/ReadVariableOp<^transformer_block/sequential/dense/BiasAdd_4/ReadVariableOp<^transformer_block/sequential/dense/Tensordot/ReadVariableOp>^transformer_block/sequential/dense/Tensordot_1/ReadVariableOp>^transformer_block/sequential/dense/Tensordot_2/ReadVariableOp>^transformer_block/sequential/dense/Tensordot_3/ReadVariableOp>^transformer_block/sequential/dense/Tensordot_4/ReadVariableOp<^transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp>^transformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOp>^transformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOp>^transformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOp>^transformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOp>^transformer_block/sequential/dense_1/Tensordot/ReadVariableOp@^transformer_block/sequential/dense_1/Tensordot_1/ReadVariableOp@^transformer_block/sequential/dense_1/Tensordot_2/ReadVariableOp@^transformer_block/sequential/dense_1/Tensordot_3/ReadVariableOp@^transformer_block/sequential/dense_1/Tensordot_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������-: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2D
 dense_6/Tensordot/ReadVariableOp dense_6/Tensordot/ReadVariableOp2r
7token_and_position_embedding/embedding/embedding_lookup7token_and_position_embedding/embedding/embedding_lookup2v
9token_and_position_embedding/embedding_1/embedding_lookup9token_and_position_embedding/embedding_1/embedding_lookup2�
>transformer_block/layer_normalization/batchnorm/ReadVariableOp>transformer_block/layer_normalization/batchnorm/ReadVariableOp2�
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpBtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2�
@transformer_block/layer_normalization/batchnorm_1/ReadVariableOp@transformer_block/layer_normalization/batchnorm_1/ReadVariableOp2�
Dtransformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOpDtransformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOp2�
@transformer_block/layer_normalization/batchnorm_2/ReadVariableOp@transformer_block/layer_normalization/batchnorm_2/ReadVariableOp2�
Dtransformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOpDtransformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOp2�
@transformer_block/layer_normalization/batchnorm_3/ReadVariableOp@transformer_block/layer_normalization/batchnorm_3/ReadVariableOp2�
Dtransformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOpDtransformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOp2�
@transformer_block/layer_normalization/batchnorm_4/ReadVariableOp@transformer_block/layer_normalization/batchnorm_4/ReadVariableOp2�
Dtransformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOpDtransformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOp2�
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2�
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpDtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2�
Btransformer_block/layer_normalization_1/batchnorm_1/ReadVariableOpBtransformer_block/layer_normalization_1/batchnorm_1/ReadVariableOp2�
Ftransformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOpFtransformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOp2�
Btransformer_block/layer_normalization_1/batchnorm_2/ReadVariableOpBtransformer_block/layer_normalization_1/batchnorm_2/ReadVariableOp2�
Ftransformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOpFtransformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOp2�
Btransformer_block/layer_normalization_1/batchnorm_3/ReadVariableOpBtransformer_block/layer_normalization_1/batchnorm_3/ReadVariableOp2�
Ftransformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOpFtransformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOp2�
Btransformer_block/layer_normalization_1/batchnorm_4/ReadVariableOpBtransformer_block/layer_normalization_1/batchnorm_4/ReadVariableOp2�
Ftransformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOpFtransformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOp2�
Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOpJtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp2�
Ltransformer_block/multi_head_attention/attention_output/add_1/ReadVariableOpLtransformer_block/multi_head_attention/attention_output/add_1/ReadVariableOp2�
Ltransformer_block/multi_head_attention/attention_output/add_2/ReadVariableOpLtransformer_block/multi_head_attention/attention_output/add_2/ReadVariableOp2�
Ltransformer_block/multi_head_attention/attention_output/add_3/ReadVariableOpLtransformer_block/multi_head_attention/attention_output/add_3/ReadVariableOp2�
Ltransformer_block/multi_head_attention/attention_output/add_4/ReadVariableOpLtransformer_block/multi_head_attention/attention_output/add_4/ReadVariableOp2�
Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpTtransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2�
Vtransformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOpVtransformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOp2�
Vtransformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOpVtransformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOp2�
Vtransformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOpVtransformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOp2�
Vtransformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOpVtransformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOp2~
=transformer_block/multi_head_attention/key/add/ReadVariableOp=transformer_block/multi_head_attention/key/add/ReadVariableOp2�
?transformer_block/multi_head_attention/key/add_1/ReadVariableOp?transformer_block/multi_head_attention/key/add_1/ReadVariableOp2�
?transformer_block/multi_head_attention/key/add_2/ReadVariableOp?transformer_block/multi_head_attention/key/add_2/ReadVariableOp2�
?transformer_block/multi_head_attention/key/add_3/ReadVariableOp?transformer_block/multi_head_attention/key/add_3/ReadVariableOp2�
?transformer_block/multi_head_attention/key/add_4/ReadVariableOp?transformer_block/multi_head_attention/key/add_4/ReadVariableOp2�
Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpGtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp2�
Itransformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOpItransformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOp2�
Itransformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOpItransformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOp2�
Itransformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOpItransformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOp2�
Itransformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOpItransformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOp2�
?transformer_block/multi_head_attention/query/add/ReadVariableOp?transformer_block/multi_head_attention/query/add/ReadVariableOp2�
Atransformer_block/multi_head_attention/query/add_1/ReadVariableOpAtransformer_block/multi_head_attention/query/add_1/ReadVariableOp2�
Atransformer_block/multi_head_attention/query/add_2/ReadVariableOpAtransformer_block/multi_head_attention/query/add_2/ReadVariableOp2�
Atransformer_block/multi_head_attention/query/add_3/ReadVariableOpAtransformer_block/multi_head_attention/query/add_3/ReadVariableOp2�
Atransformer_block/multi_head_attention/query/add_4/ReadVariableOpAtransformer_block/multi_head_attention/query/add_4/ReadVariableOp2�
Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpItransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp2�
Ktransformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOpKtransformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOp2�
Ktransformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOpKtransformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOp2�
Ktransformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOpKtransformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOp2�
Ktransformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOpKtransformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOp2�
?transformer_block/multi_head_attention/value/add/ReadVariableOp?transformer_block/multi_head_attention/value/add/ReadVariableOp2�
Atransformer_block/multi_head_attention/value/add_1/ReadVariableOpAtransformer_block/multi_head_attention/value/add_1/ReadVariableOp2�
Atransformer_block/multi_head_attention/value/add_2/ReadVariableOpAtransformer_block/multi_head_attention/value/add_2/ReadVariableOp2�
Atransformer_block/multi_head_attention/value/add_3/ReadVariableOpAtransformer_block/multi_head_attention/value/add_3/ReadVariableOp2�
Atransformer_block/multi_head_attention/value/add_4/ReadVariableOpAtransformer_block/multi_head_attention/value/add_4/ReadVariableOp2�
Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpItransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp2�
Ktransformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOpKtransformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOp2�
Ktransformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOpKtransformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOp2�
Ktransformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOpKtransformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOp2�
Ktransformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOpKtransformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOp2v
9transformer_block/sequential/dense/BiasAdd/ReadVariableOp9transformer_block/sequential/dense/BiasAdd/ReadVariableOp2z
;transformer_block/sequential/dense/BiasAdd_1/ReadVariableOp;transformer_block/sequential/dense/BiasAdd_1/ReadVariableOp2z
;transformer_block/sequential/dense/BiasAdd_2/ReadVariableOp;transformer_block/sequential/dense/BiasAdd_2/ReadVariableOp2z
;transformer_block/sequential/dense/BiasAdd_3/ReadVariableOp;transformer_block/sequential/dense/BiasAdd_3/ReadVariableOp2z
;transformer_block/sequential/dense/BiasAdd_4/ReadVariableOp;transformer_block/sequential/dense/BiasAdd_4/ReadVariableOp2z
;transformer_block/sequential/dense/Tensordot/ReadVariableOp;transformer_block/sequential/dense/Tensordot/ReadVariableOp2~
=transformer_block/sequential/dense/Tensordot_1/ReadVariableOp=transformer_block/sequential/dense/Tensordot_1/ReadVariableOp2~
=transformer_block/sequential/dense/Tensordot_2/ReadVariableOp=transformer_block/sequential/dense/Tensordot_2/ReadVariableOp2~
=transformer_block/sequential/dense/Tensordot_3/ReadVariableOp=transformer_block/sequential/dense/Tensordot_3/ReadVariableOp2~
=transformer_block/sequential/dense/Tensordot_4/ReadVariableOp=transformer_block/sequential/dense/Tensordot_4/ReadVariableOp2z
;transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp;transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp2~
=transformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOp=transformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOp2~
=transformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOp=transformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOp2~
=transformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOp=transformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOp2~
=transformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOp=transformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOp2~
=transformer_block/sequential/dense_1/Tensordot/ReadVariableOp=transformer_block/sequential/dense_1/Tensordot/ReadVariableOp2�
?transformer_block/sequential/dense_1/Tensordot_1/ReadVariableOp?transformer_block/sequential/dense_1/Tensordot_1/ReadVariableOp2�
?transformer_block/sequential/dense_1/Tensordot_2/ReadVariableOp?transformer_block/sequential/dense_1/Tensordot_2/ReadVariableOp2�
?transformer_block/sequential/dense_1/Tensordot_3/ReadVariableOp?transformer_block/sequential/dense_1/Tensordot_3/ReadVariableOp2�
?transformer_block/sequential/dense_1/Tensordot_4/ReadVariableOp?transformer_block/sequential/dense_1/Tensordot_4/ReadVariableOp:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_2370

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������-`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������-"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������-:T P
,
_output_shapes
:����������-
 
_user_specified_nameinputs
�
a
C__inference_dropout_4_layer_call_and_return_conditional_losses_2458

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

b
C__inference_dropout_2_layer_call_and_return_conditional_losses_2807

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������-C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������-*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * gL=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������-t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������-n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������-^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������-:T P
,
_output_shapes
:����������-
 
_user_specified_nameinputs
�
�
&__inference_dense_4_layer_call_fn_5942

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_2491s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
C__inference_dropout_3_layer_call_and_return_conditional_losses_5854

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_dense_1_layer_call_fn_6296

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1965t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������-�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������-�: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_3832

inputs
unknown:	-�
	unknown_0:
��!
	unknown_1:��
	unknown_2:	�!
	unknown_3:��
	unknown_4:	�!
	unknown_5:��
	unknown_6:	�!
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
�-�

unknown_18:	�

unknown_19:
��

unknown_20:	�

unknown_21:	�@

unknown_22:@

unknown_23:@

unknown_24:

unknown_25:

unknown_26:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*>
_read_only_resource_inputs 
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_3245s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������-: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�
�
&__inference_dense_6_layer_call_fn_6076

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_2579s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
A__inference_dense_5_layer_call_and_return_conditional_losses_6040

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������@�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
SeluSeluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentitySelu:activations:0^NoOp*
T0*+
_output_shapes
:���������z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
a
C__inference_dropout_6_layer_call_and_return_conditional_losses_6055

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
a
(__inference_dropout_6_layer_call_fn_6050

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_6_layer_call_and_return_conditional_losses_2675s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
a
C__inference_dropout_4_layer_call_and_return_conditional_losses_5921

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�=
�
D__inference_sequential_layer_call_and_return_conditional_losses_6190

inputs;
'dense_tensordot_readvariableop_resource:
��4
%dense_biasadd_readvariableop_resource:	�=
)dense_1_tensordot_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       K
dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�a

dense/SeluSeludense/BiasAdd:output:0*
T0*,
_output_shapes
:���������-��
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapedense/Selu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_1/Tensordot/transpose	Transposedense/Selu:activations:0!dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-��
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�l
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������-��
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������-�: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�
�
A__inference_dense_6_layer_call_and_return_conditional_losses_6107

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
SeluSeluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentitySelu:activations:0^NoOp*
T0*+
_output_shapes
:���������z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
a
(__inference_dropout_5_layer_call_fn_5983

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_5_layer_call_and_return_conditional_losses_2708s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
A__inference_dense_2_layer_call_and_return_conditional_losses_5839

inputs5
!tensordot_readvariableop_resource:
�-�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
�-�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:����������-�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
SeluSeluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentitySelu:activations:0^NoOp*
T0*,
_output_shapes
:����������z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������-
 
_user_specified_nameinputs
�

b
C__inference_dropout_3_layer_call_and_return_conditional_losses_2774

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * gL=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

b
C__inference_dropout_4_layer_call_and_return_conditional_losses_2741

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * gL=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�g
�
?__inference_model_layer_call_and_return_conditional_losses_3647
input_14
!token_and_position_embedding_3509:	-�5
!token_and_position_embedding_3511:
��.
transformer_block_3514:��)
transformer_block_3516:	�.
transformer_block_3518:��)
transformer_block_3520:	�.
transformer_block_3522:��)
transformer_block_3524:	�.
transformer_block_3526:��%
transformer_block_3528:	�%
transformer_block_3530:	�%
transformer_block_3532:	�*
transformer_block_3534:
��%
transformer_block_3536:	�*
transformer_block_3538:
��%
transformer_block_3540:	�%
transformer_block_3542:	�%
transformer_block_3544:	� 
dense_2_3617:
�-�
dense_2_3619:	� 
dense_3_3623:
��
dense_3_3625:	�
dense_4_3629:	�@
dense_4_3631:@
dense_5_3635:@
dense_5_3637:
dense_6_3641:
dense_6_3643:
identity��dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�!dropout_3/StatefulPartitionedCall�!dropout_4/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCall�!dropout_6/StatefulPartitionedCall�4token_and_position_embedding/StatefulPartitionedCall�)transformer_block/StatefulPartitionedCall�+transformer_block/StatefulPartitionedCall_1�+transformer_block/StatefulPartitionedCall_2�+transformer_block/StatefulPartitionedCall_3�+transformer_block/StatefulPartitionedCall_4�
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1!token_and_position_embedding_3509!token_and_position_embedding_3511*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *_
fZRX
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_2115�
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_block_3514transformer_block_3516transformer_block_3518transformer_block_3520transformer_block_3522transformer_block_3524transformer_block_3526transformer_block_3528transformer_block_3530transformer_block_3532transformer_block_3534transformer_block_3536transformer_block_3538transformer_block_3540transformer_block_3542transformer_block_3544*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2996�
+transformer_block/StatefulPartitionedCall_1StatefulPartitionedCall2transformer_block/StatefulPartitionedCall:output:0transformer_block_3514transformer_block_3516transformer_block_3518transformer_block_3520transformer_block_3522transformer_block_3524transformer_block_3526transformer_block_3528transformer_block_3530transformer_block_3532transformer_block_3534transformer_block_3536transformer_block_3538transformer_block_3540transformer_block_3542transformer_block_3544*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2996�
+transformer_block/StatefulPartitionedCall_2StatefulPartitionedCall4transformer_block/StatefulPartitionedCall_1:output:0transformer_block_3514transformer_block_3516transformer_block_3518transformer_block_3520transformer_block_3522transformer_block_3524transformer_block_3526transformer_block_3528transformer_block_3530transformer_block_3532transformer_block_3534transformer_block_3536transformer_block_3538transformer_block_3540transformer_block_3542transformer_block_3544*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2996�
+transformer_block/StatefulPartitionedCall_3StatefulPartitionedCall4transformer_block/StatefulPartitionedCall_2:output:0transformer_block_3514transformer_block_3516transformer_block_3518transformer_block_3520transformer_block_3522transformer_block_3524transformer_block_3526transformer_block_3528transformer_block_3530transformer_block_3532transformer_block_3534transformer_block_3536transformer_block_3538transformer_block_3540transformer_block_3542transformer_block_3544*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2996�
+transformer_block/StatefulPartitionedCall_4StatefulPartitionedCall4transformer_block/StatefulPartitionedCall_3:output:0transformer_block_3514transformer_block_3516transformer_block_3518transformer_block_3520transformer_block_3522transformer_block_3524transformer_block_3526transformer_block_3528transformer_block_3530transformer_block_3532transformer_block_3534transformer_block_3536transformer_block_3538transformer_block_3540transformer_block_3542transformer_block_3544*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2996�
reshape/PartitionedCallPartitionedCall4transformer_block/StatefulPartitionedCall_4:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_2363�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_2807�
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_3617dense_2_3619*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_2403�
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_2774�
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_3_3623dense_3_3625*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_2447�
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_2741�
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_4_3629dense_4_3631*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_2491�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_5_layer_call_and_return_conditional_losses_2708�
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_5_3635dense_5_3637*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_2535�
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0"^dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_6_layer_call_and_return_conditional_losses_2675�
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0dense_6_3641dense_6_3643*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_2579{
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall,^transformer_block/StatefulPartitionedCall_1,^transformer_block/StatefulPartitionedCall_2,^transformer_block/StatefulPartitionedCall_3,^transformer_block/StatefulPartitionedCall_4*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������-: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall2Z
+transformer_block/StatefulPartitionedCall_1+transformer_block/StatefulPartitionedCall_12Z
+transformer_block/StatefulPartitionedCall_2+transformer_block/StatefulPartitionedCall_22Z
+transformer_block/StatefulPartitionedCall_3+transformer_block/StatefulPartitionedCall_32Z
+transformer_block/StatefulPartitionedCall_4+transformer_block/StatefulPartitionedCall_4:P L
'
_output_shapes
:���������-
!
_user_specified_name	input_1
�
a
(__inference_dropout_3_layer_call_fn_5849

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_2774t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
��
�B
?__inference_model_layer_call_and_return_conditional_losses_5380

inputsQ
>token_and_position_embedding_embedding_1_embedding_lookup_4567:	-�P
<token_and_position_embedding_embedding_embedding_lookup_4573:
��j
Rtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource:��[
Htransformer_block_multi_head_attention_query_add_readvariableop_resource:	�h
Ptransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource:��Y
Ftransformer_block_multi_head_attention_key_add_readvariableop_resource:	�j
Rtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource:��[
Htransformer_block_multi_head_attention_value_add_readvariableop_resource:	�u
]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:��b
Stransformer_block_multi_head_attention_attention_output_add_readvariableop_resource:	�Z
Ktransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource:	�V
Gtransformer_block_layer_normalization_batchnorm_readvariableop_resource:	�X
Dtransformer_block_sequential_dense_tensordot_readvariableop_resource:
��Q
Btransformer_block_sequential_dense_biasadd_readvariableop_resource:	�Z
Ftransformer_block_sequential_dense_1_tensordot_readvariableop_resource:
��S
Dtransformer_block_sequential_dense_1_biasadd_readvariableop_resource:	�\
Mtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource:	�X
Itransformer_block_layer_normalization_1_batchnorm_readvariableop_resource:	�=
)dense_2_tensordot_readvariableop_resource:
�-�6
'dense_2_biasadd_readvariableop_resource:	�=
)dense_3_tensordot_readvariableop_resource:
��6
'dense_3_biasadd_readvariableop_resource:	�<
)dense_4_tensordot_readvariableop_resource:	�@5
'dense_4_biasadd_readvariableop_resource:@;
)dense_5_tensordot_readvariableop_resource:@5
'dense_5_biasadd_readvariableop_resource:;
)dense_6_tensordot_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:
identity��dense_2/BiasAdd/ReadVariableOp� dense_2/Tensordot/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp� dense_3/Tensordot/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp� dense_4/Tensordot/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp� dense_5/Tensordot/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp� dense_6/Tensordot/ReadVariableOp�7token_and_position_embedding/embedding/embedding_lookup�9token_and_position_embedding/embedding_1/embedding_lookup�>transformer_block/layer_normalization/batchnorm/ReadVariableOp�Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp�@transformer_block/layer_normalization/batchnorm_1/ReadVariableOp�Dtransformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOp�@transformer_block/layer_normalization/batchnorm_2/ReadVariableOp�Dtransformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOp�@transformer_block/layer_normalization/batchnorm_3/ReadVariableOp�Dtransformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOp�@transformer_block/layer_normalization/batchnorm_4/ReadVariableOp�Dtransformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOp�@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp�Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp�Btransformer_block/layer_normalization_1/batchnorm_1/ReadVariableOp�Ftransformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOp�Btransformer_block/layer_normalization_1/batchnorm_2/ReadVariableOp�Ftransformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOp�Btransformer_block/layer_normalization_1/batchnorm_3/ReadVariableOp�Ftransformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOp�Btransformer_block/layer_normalization_1/batchnorm_4/ReadVariableOp�Ftransformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOp�Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp�Ltransformer_block/multi_head_attention/attention_output/add_1/ReadVariableOp�Ltransformer_block/multi_head_attention/attention_output/add_2/ReadVariableOp�Ltransformer_block/multi_head_attention/attention_output/add_3/ReadVariableOp�Ltransformer_block/multi_head_attention/attention_output/add_4/ReadVariableOp�Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp�Vtransformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOp�Vtransformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOp�Vtransformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOp�Vtransformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOp�=transformer_block/multi_head_attention/key/add/ReadVariableOp�?transformer_block/multi_head_attention/key/add_1/ReadVariableOp�?transformer_block/multi_head_attention/key/add_2/ReadVariableOp�?transformer_block/multi_head_attention/key/add_3/ReadVariableOp�?transformer_block/multi_head_attention/key/add_4/ReadVariableOp�Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp�Itransformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOp�Itransformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOp�Itransformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOp�Itransformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOp�?transformer_block/multi_head_attention/query/add/ReadVariableOp�Atransformer_block/multi_head_attention/query/add_1/ReadVariableOp�Atransformer_block/multi_head_attention/query/add_2/ReadVariableOp�Atransformer_block/multi_head_attention/query/add_3/ReadVariableOp�Atransformer_block/multi_head_attention/query/add_4/ReadVariableOp�Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp�Ktransformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOp�Ktransformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOp�Ktransformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOp�Ktransformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOp�?transformer_block/multi_head_attention/value/add/ReadVariableOp�Atransformer_block/multi_head_attention/value/add_1/ReadVariableOp�Atransformer_block/multi_head_attention/value/add_2/ReadVariableOp�Atransformer_block/multi_head_attention/value/add_3/ReadVariableOp�Atransformer_block/multi_head_attention/value/add_4/ReadVariableOp�Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp�Ktransformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOp�Ktransformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOp�Ktransformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOp�Ktransformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOp�9transformer_block/sequential/dense/BiasAdd/ReadVariableOp�;transformer_block/sequential/dense/BiasAdd_1/ReadVariableOp�;transformer_block/sequential/dense/BiasAdd_2/ReadVariableOp�;transformer_block/sequential/dense/BiasAdd_3/ReadVariableOp�;transformer_block/sequential/dense/BiasAdd_4/ReadVariableOp�;transformer_block/sequential/dense/Tensordot/ReadVariableOp�=transformer_block/sequential/dense/Tensordot_1/ReadVariableOp�=transformer_block/sequential/dense/Tensordot_2/ReadVariableOp�=transformer_block/sequential/dense/Tensordot_3/ReadVariableOp�=transformer_block/sequential/dense/Tensordot_4/ReadVariableOp�;transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp�=transformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOp�=transformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOp�=transformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOp�=transformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOp�=transformer_block/sequential/dense_1/Tensordot/ReadVariableOp�?transformer_block/sequential/dense_1/Tensordot_1/ReadVariableOp�?transformer_block/sequential/dense_1/Tensordot_2/ReadVariableOp�?transformer_block/sequential/dense_1/Tensordot_3/ReadVariableOp�?transformer_block/sequential/dense_1/Tensordot_4/ReadVariableOpX
"token_and_position_embedding/ShapeShapeinputs*
T0*
_output_shapes
:�
0token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������|
2token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*token_and_position_embedding/strided_sliceStridedSlice+token_and_position_embedding/Shape:output:09token_and_position_embedding/strided_slice/stack:output:0;token_and_position_embedding/strided_slice/stack_1:output:0;token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : j
(token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
"token_and_position_embedding/rangeRange1token_and_position_embedding/range/start:output:03token_and_position_embedding/strided_slice:output:01token_and_position_embedding/range/delta:output:0*
_output_shapes
:-�
9token_and_position_embedding/embedding_1/embedding_lookupResourceGather>token_and_position_embedding_embedding_1_embedding_lookup_4567+token_and_position_embedding/range:output:0*
Tindices0*Q
_classG
ECloc:@token_and_position_embedding/embedding_1/embedding_lookup/4567*
_output_shapes
:	-�*
dtype0�
Btoken_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityBtoken_and_position_embedding/embedding_1/embedding_lookup:output:0*
T0*Q
_classG
ECloc:@token_and_position_embedding/embedding_1/embedding_lookup/4567*
_output_shapes
:	-��
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityKtoken_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	-�|
+token_and_position_embedding/embedding/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������-�
7token_and_position_embedding/embedding/embedding_lookupResourceGather<token_and_position_embedding_embedding_embedding_lookup_4573/token_and_position_embedding/embedding/Cast:y:0*
Tindices0*O
_classE
CAloc:@token_and_position_embedding/embedding/embedding_lookup/4573*,
_output_shapes
:���������-�*
dtype0�
@token_and_position_embedding/embedding/embedding_lookup/IdentityIdentity@token_and_position_embedding/embedding/embedding_lookup:output:0*
T0*O
_classE
CAloc:@token_and_position_embedding/embedding/embedding_lookup/4573*,
_output_shapes
:���������-��
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityItoken_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:���������-��
 token_and_position_embedding/addAddV2Ktoken_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Mtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:���������-��
Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
:transformer_block/multi_head_attention/query/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Qtransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
?transformer_block/multi_head_attention/query/add/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
0transformer_block/multi_head_attention/query/addAddV2Ctransformer_block/multi_head_attention/query/einsum/Einsum:output:0Gtransformer_block/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpPtransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
8transformer_block/multi_head_attention/key/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Otransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
=transformer_block/multi_head_attention/key/add/ReadVariableOpReadVariableOpFtransformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
.transformer_block/multi_head_attention/key/addAddV2Atransformer_block/multi_head_attention/key/einsum/Einsum:output:0Etransformer_block/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
:transformer_block/multi_head_attention/value/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Qtransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
?transformer_block/multi_head_attention/value/add/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
0transformer_block/multi_head_attention/value/addAddV2Ctransformer_block/multi_head_attention/value/einsum/Einsum:output:0Gtransformer_block/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�q
,transformer_block/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
*transformer_block/multi_head_attention/MulMul4transformer_block/multi_head_attention/query/add:z:05transformer_block/multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:���������-��
4transformer_block/multi_head_attention/einsum/EinsumEinsum2transformer_block/multi_head_attention/key/add:z:0.transformer_block/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
6transformer_block/multi_head_attention/softmax/SoftmaxSoftmax=transformer_block/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������--�
6transformer_block/multi_head_attention/einsum_1/EinsumEinsum@transformer_block/multi_head_attention/softmax/Softmax:softmax:04transformer_block/multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Etransformer_block/multi_head_attention/attention_output/einsum/EinsumEinsum?transformer_block/multi_head_attention/einsum_1/Einsum:output:0\transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpStransformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;transformer_block/multi_head_attention/attention_output/addAddV2Ntransformer_block/multi_head_attention/attention_output/einsum/Einsum:output:0Rtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�l
'transformer_block/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
%transformer_block/dropout/dropout/MulMul?transformer_block/multi_head_attention/attention_output/add:z:00transformer_block/dropout/dropout/Const:output:0*
T0*,
_output_shapes
:���������-��
'transformer_block/dropout/dropout/ShapeShape?transformer_block/multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:�
>transformer_block/dropout/dropout/random_uniform/RandomUniformRandomUniform0transformer_block/dropout/dropout/Shape:output:0*
T0*,
_output_shapes
:���������-�*
dtype0u
0transformer_block/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *U;>�
.transformer_block/dropout/dropout/GreaterEqualGreaterEqualGtransformer_block/dropout/dropout/random_uniform/RandomUniform:output:09transformer_block/dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������-��
&transformer_block/dropout/dropout/CastCast2transformer_block/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������-��
'transformer_block/dropout/dropout/Mul_1Mul)transformer_block/dropout/dropout/Mul:z:0*transformer_block/dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:���������-��
transformer_block/addAddV2$token_and_position_embedding/add:z:0+transformer_block/dropout/dropout/Mul_1:z:0*
T0*,
_output_shapes
:���������-��
Dtransformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
2transformer_block/layer_normalization/moments/meanMeantransformer_block/add:z:0Mtransformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
:transformer_block/layer_normalization/moments/StopGradientStopGradient;transformer_block/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:���������-�
?transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_block/add:z:0Ctransformer_block/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Htransformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
6transformer_block/layer_normalization/moments/varianceMeanCtransformer_block/layer_normalization/moments/SquaredDifference:z:0Qtransformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(z
5transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
3transformer_block/layer_normalization/batchnorm/addAddV2?transformer_block/layer_normalization/moments/variance:output:0>transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������-�
5transformer_block/layer_normalization/batchnorm/RsqrtRsqrt7transformer_block/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:���������-�
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
3transformer_block/layer_normalization/batchnorm/mulMul9transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Jtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
5transformer_block/layer_normalization/batchnorm/mul_1Multransformer_block/add:z:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
5transformer_block/layer_normalization/batchnorm/mul_2Mul;transformer_block/layer_normalization/moments/mean:output:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
>transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
3transformer_block/layer_normalization/batchnorm/subSubFtransformer_block/layer_normalization/batchnorm/ReadVariableOp:value:09transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������-��
5transformer_block/layer_normalization/batchnorm/add_1AddV29transformer_block/layer_normalization/batchnorm/mul_1:z:07transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������-��
;transformer_block/sequential/dense/Tensordot/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0{
1transformer_block/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
1transformer_block/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
2transformer_block/sequential/dense/Tensordot/ShapeShape9transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:|
:transformer_block/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5transformer_block/sequential/dense/Tensordot/GatherV2GatherV2;transformer_block/sequential/dense/Tensordot/Shape:output:0:transformer_block/sequential/dense/Tensordot/free:output:0Ctransformer_block/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
<transformer_block/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot/GatherV2_1GatherV2;transformer_block/sequential/dense/Tensordot/Shape:output:0:transformer_block/sequential/dense/Tensordot/axes:output:0Etransformer_block/sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:|
2transformer_block/sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
1transformer_block/sequential/dense/Tensordot/ProdProd>transformer_block/sequential/dense/Tensordot/GatherV2:output:0;transformer_block/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: ~
4transformer_block/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
3transformer_block/sequential/dense/Tensordot/Prod_1Prod@transformer_block/sequential/dense/Tensordot/GatherV2_1:output:0=transformer_block/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: z
8transformer_block/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
3transformer_block/sequential/dense/Tensordot/concatConcatV2:transformer_block/sequential/dense/Tensordot/free:output:0:transformer_block/sequential/dense/Tensordot/axes:output:0Atransformer_block/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
2transformer_block/sequential/dense/Tensordot/stackPack:transformer_block/sequential/dense/Tensordot/Prod:output:0<transformer_block/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
6transformer_block/sequential/dense/Tensordot/transpose	Transpose9transformer_block/layer_normalization/batchnorm/add_1:z:0<transformer_block/sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
4transformer_block/sequential/dense/Tensordot/ReshapeReshape:transformer_block/sequential/dense/Tensordot/transpose:y:0;transformer_block/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
3transformer_block/sequential/dense/Tensordot/MatMulMatMul=transformer_block/sequential/dense/Tensordot/Reshape:output:0Ctransformer_block/sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
4transformer_block/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�|
:transformer_block/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5transformer_block/sequential/dense/Tensordot/concat_1ConcatV2>transformer_block/sequential/dense/Tensordot/GatherV2:output:0=transformer_block/sequential/dense/Tensordot/Const_2:output:0Ctransformer_block/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
,transformer_block/sequential/dense/TensordotReshape=transformer_block/sequential/dense/Tensordot/MatMul:product:0>transformer_block/sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-��
9transformer_block/sequential/dense/BiasAdd/ReadVariableOpReadVariableOpBtransformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*transformer_block/sequential/dense/BiasAddBiasAdd5transformer_block/sequential/dense/Tensordot:output:0Atransformer_block/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
'transformer_block/sequential/dense/SeluSelu3transformer_block/sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense_1/Tensordot/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0}
3transformer_block/sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
3transformer_block/sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
4transformer_block/sequential/dense_1/Tensordot/ShapeShape5transformer_block/sequential/dense/Selu:activations:0*
T0*
_output_shapes
:~
<transformer_block/sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense_1/Tensordot/GatherV2GatherV2=transformer_block/sequential/dense_1/Tensordot/Shape:output:0<transformer_block/sequential/dense_1/Tensordot/free:output:0Etransformer_block/sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
>transformer_block/sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot/GatherV2_1GatherV2=transformer_block/sequential/dense_1/Tensordot/Shape:output:0<transformer_block/sequential/dense_1/Tensordot/axes:output:0Gtransformer_block/sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4transformer_block/sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
3transformer_block/sequential/dense_1/Tensordot/ProdProd@transformer_block/sequential/dense_1/Tensordot/GatherV2:output:0=transformer_block/sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
6transformer_block/sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense_1/Tensordot/Prod_1ProdBtransformer_block/sequential/dense_1/Tensordot/GatherV2_1:output:0?transformer_block/sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_block/sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5transformer_block/sequential/dense_1/Tensordot/concatConcatV2<transformer_block/sequential/dense_1/Tensordot/free:output:0<transformer_block/sequential/dense_1/Tensordot/axes:output:0Ctransformer_block/sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
4transformer_block/sequential/dense_1/Tensordot/stackPack<transformer_block/sequential/dense_1/Tensordot/Prod:output:0>transformer_block/sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
8transformer_block/sequential/dense_1/Tensordot/transpose	Transpose5transformer_block/sequential/dense/Selu:activations:0>transformer_block/sequential/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
6transformer_block/sequential/dense_1/Tensordot/ReshapeReshape<transformer_block/sequential/dense_1/Tensordot/transpose:y:0=transformer_block/sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
5transformer_block/sequential/dense_1/Tensordot/MatMulMatMul?transformer_block/sequential/dense_1/Tensordot/Reshape:output:0Etransformer_block/sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
6transformer_block/sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�~
<transformer_block/sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense_1/Tensordot/concat_1ConcatV2@transformer_block/sequential/dense_1/Tensordot/GatherV2:output:0?transformer_block/sequential/dense_1/Tensordot/Const_2:output:0Etransformer_block/sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
.transformer_block/sequential/dense_1/TensordotReshape?transformer_block/sequential/dense_1/Tensordot/MatMul:product:0@transformer_block/sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-��
;transformer_block/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,transformer_block/sequential/dense_1/BiasAddBiasAdd7transformer_block/sequential/dense_1/Tensordot:output:0Ctransformer_block/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�n
)transformer_block/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
'transformer_block/dropout_1/dropout/MulMul5transformer_block/sequential/dense_1/BiasAdd:output:02transformer_block/dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:���������-��
)transformer_block/dropout_1/dropout/ShapeShape5transformer_block/sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:�
@transformer_block/dropout_1/dropout/random_uniform/RandomUniformRandomUniform2transformer_block/dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:���������-�*
dtype0w
2transformer_block/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *U;>�
0transformer_block/dropout_1/dropout/GreaterEqualGreaterEqualItransformer_block/dropout_1/dropout/random_uniform/RandomUniform:output:0;transformer_block/dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������-��
(transformer_block/dropout_1/dropout/CastCast4transformer_block/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������-��
)transformer_block/dropout_1/dropout/Mul_1Mul+transformer_block/dropout_1/dropout/Mul:z:0,transformer_block/dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_1AddV29transformer_block/layer_normalization/batchnorm/add_1:z:0-transformer_block/dropout_1/dropout/Mul_1:z:0*
T0*,
_output_shapes
:���������-��
Ftransformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
4transformer_block/layer_normalization_1/moments/meanMeantransformer_block/add_1:z:0Otransformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
<transformer_block/layer_normalization_1/moments/StopGradientStopGradient=transformer_block/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:���������-�
Atransformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_block/add_1:z:0Etransformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Jtransformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
8transformer_block/layer_normalization_1/moments/varianceMeanEtransformer_block/layer_normalization_1/moments/SquaredDifference:z:0Stransformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(|
7transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
5transformer_block/layer_normalization_1/batchnorm/addAddV2Atransformer_block/layer_normalization_1/moments/variance:output:0@transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������-�
7transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt9transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:���������-�
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization_1/batchnorm/mulMul;transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Ltransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization_1/batchnorm/mul_1Multransformer_block/add_1:z:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization_1/batchnorm/mul_2Mul=transformer_block/layer_normalization_1/moments/mean:output:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization_1/batchnorm/subSubHtransformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0;transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization_1/batchnorm/add_1AddV2;transformer_block/layer_normalization_1/batchnorm/mul_1:z:09transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������-��
Ktransformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_block/multi_head_attention/query/einsum_1/EinsumEinsum;transformer_block/layer_normalization_1/batchnorm/add_1:z:0Stransformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Atransformer_block/multi_head_attention/query/add_1/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_block/multi_head_attention/query/add_1AddV2Etransformer_block/multi_head_attention/query/einsum_1/Einsum:output:0Itransformer_block/multi_head_attention/query/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Itransformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOpReadVariableOpPtransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
:transformer_block/multi_head_attention/key/einsum_1/EinsumEinsum;transformer_block/layer_normalization_1/batchnorm/add_1:z:0Qtransformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
?transformer_block/multi_head_attention/key/add_1/ReadVariableOpReadVariableOpFtransformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
0transformer_block/multi_head_attention/key/add_1AddV2Ctransformer_block/multi_head_attention/key/einsum_1/Einsum:output:0Gtransformer_block/multi_head_attention/key/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Ktransformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_block/multi_head_attention/value/einsum_1/EinsumEinsum;transformer_block/layer_normalization_1/batchnorm/add_1:z:0Stransformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Atransformer_block/multi_head_attention/value/add_1/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_block/multi_head_attention/value/add_1AddV2Etransformer_block/multi_head_attention/value/einsum_1/Einsum:output:0Itransformer_block/multi_head_attention/value/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�s
.transformer_block/multi_head_attention/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
,transformer_block/multi_head_attention/Mul_1Mul6transformer_block/multi_head_attention/query/add_1:z:07transformer_block/multi_head_attention/Mul_1/y:output:0*
T0*0
_output_shapes
:���������-��
6transformer_block/multi_head_attention/einsum_2/EinsumEinsum4transformer_block/multi_head_attention/key/add_1:z:00transformer_block/multi_head_attention/Mul_1:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
8transformer_block/multi_head_attention/softmax/Softmax_1Softmax?transformer_block/multi_head_attention/einsum_2/Einsum:output:0*
T0*/
_output_shapes
:���������--�
6transformer_block/multi_head_attention/einsum_3/EinsumEinsumBtransformer_block/multi_head_attention/softmax/Softmax_1:softmax:06transformer_block/multi_head_attention/value/add_1:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
Vtransformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOpReadVariableOp]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Gtransformer_block/multi_head_attention/attention_output/einsum_1/EinsumEinsum?transformer_block/multi_head_attention/einsum_3/Einsum:output:0^transformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
Ltransformer_block/multi_head_attention/attention_output/add_1/ReadVariableOpReadVariableOpStransformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=transformer_block/multi_head_attention/attention_output/add_1AddV2Ptransformer_block/multi_head_attention/attention_output/einsum_1/Einsum:output:0Ttransformer_block/multi_head_attention/attention_output/add_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�n
)transformer_block/dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
'transformer_block/dropout/dropout_1/MulMulAtransformer_block/multi_head_attention/attention_output/add_1:z:02transformer_block/dropout/dropout_1/Const:output:0*
T0*,
_output_shapes
:���������-��
)transformer_block/dropout/dropout_1/ShapeShapeAtransformer_block/multi_head_attention/attention_output/add_1:z:0*
T0*
_output_shapes
:�
@transformer_block/dropout/dropout_1/random_uniform/RandomUniformRandomUniform2transformer_block/dropout/dropout_1/Shape:output:0*
T0*,
_output_shapes
:���������-�*
dtype0w
2transformer_block/dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *U;>�
0transformer_block/dropout/dropout_1/GreaterEqualGreaterEqualItransformer_block/dropout/dropout_1/random_uniform/RandomUniform:output:0;transformer_block/dropout/dropout_1/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������-��
(transformer_block/dropout/dropout_1/CastCast4transformer_block/dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������-��
)transformer_block/dropout/dropout_1/Mul_1Mul+transformer_block/dropout/dropout_1/Mul:z:0,transformer_block/dropout/dropout_1/Cast:y:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_2AddV2;transformer_block/layer_normalization_1/batchnorm/add_1:z:0-transformer_block/dropout/dropout_1/Mul_1:z:0*
T0*,
_output_shapes
:���������-��
Ftransformer_block/layer_normalization/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
4transformer_block/layer_normalization/moments_1/meanMeantransformer_block/add_2:z:0Otransformer_block/layer_normalization/moments_1/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
<transformer_block/layer_normalization/moments_1/StopGradientStopGradient=transformer_block/layer_normalization/moments_1/mean:output:0*
T0*+
_output_shapes
:���������-�
Atransformer_block/layer_normalization/moments_1/SquaredDifferenceSquaredDifferencetransformer_block/add_2:z:0Etransformer_block/layer_normalization/moments_1/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Jtransformer_block/layer_normalization/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
8transformer_block/layer_normalization/moments_1/varianceMeanEtransformer_block/layer_normalization/moments_1/SquaredDifference:z:0Stransformer_block/layer_normalization/moments_1/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(|
7transformer_block/layer_normalization/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
5transformer_block/layer_normalization/batchnorm_1/addAddV2Atransformer_block/layer_normalization/moments_1/variance:output:0@transformer_block/layer_normalization/batchnorm_1/add/y:output:0*
T0*+
_output_shapes
:���������-�
7transformer_block/layer_normalization/batchnorm_1/RsqrtRsqrt9transformer_block/layer_normalization/batchnorm_1/add:z:0*
T0*+
_output_shapes
:���������-�
Dtransformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization/batchnorm_1/mulMul;transformer_block/layer_normalization/batchnorm_1/Rsqrt:y:0Ltransformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_1/mul_1Multransformer_block/add_2:z:09transformer_block/layer_normalization/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_1/mul_2Mul=transformer_block/layer_normalization/moments_1/mean:output:09transformer_block/layer_normalization/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:���������-��
@transformer_block/layer_normalization/batchnorm_1/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization/batchnorm_1/subSubHtransformer_block/layer_normalization/batchnorm_1/ReadVariableOp:value:0;transformer_block/layer_normalization/batchnorm_1/mul_2:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_1/add_1AddV2;transformer_block/layer_normalization/batchnorm_1/mul_1:z:09transformer_block/layer_normalization/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense/Tensordot_1/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0}
3transformer_block/sequential/dense/Tensordot_1/axesConst*
_output_shapes
:*
dtype0*
valueB:�
3transformer_block/sequential/dense/Tensordot_1/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
4transformer_block/sequential/dense/Tensordot_1/ShapeShape;transformer_block/layer_normalization/batchnorm_1/add_1:z:0*
T0*
_output_shapes
:~
<transformer_block/sequential/dense/Tensordot_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot_1/GatherV2GatherV2=transformer_block/sequential/dense/Tensordot_1/Shape:output:0<transformer_block/sequential/dense/Tensordot_1/free:output:0Etransformer_block/sequential/dense/Tensordot_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
>transformer_block/sequential/dense/Tensordot_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense/Tensordot_1/GatherV2_1GatherV2=transformer_block/sequential/dense/Tensordot_1/Shape:output:0<transformer_block/sequential/dense/Tensordot_1/axes:output:0Gtransformer_block/sequential/dense/Tensordot_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4transformer_block/sequential/dense/Tensordot_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
3transformer_block/sequential/dense/Tensordot_1/ProdProd@transformer_block/sequential/dense/Tensordot_1/GatherV2:output:0=transformer_block/sequential/dense/Tensordot_1/Const:output:0*
T0*
_output_shapes
: �
6transformer_block/sequential/dense/Tensordot_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense/Tensordot_1/Prod_1ProdBtransformer_block/sequential/dense/Tensordot_1/GatherV2_1:output:0?transformer_block/sequential/dense/Tensordot_1/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_block/sequential/dense/Tensordot_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5transformer_block/sequential/dense/Tensordot_1/concatConcatV2<transformer_block/sequential/dense/Tensordot_1/free:output:0<transformer_block/sequential/dense/Tensordot_1/axes:output:0Ctransformer_block/sequential/dense/Tensordot_1/concat/axis:output:0*
N*
T0*
_output_shapes
:�
4transformer_block/sequential/dense/Tensordot_1/stackPack<transformer_block/sequential/dense/Tensordot_1/Prod:output:0>transformer_block/sequential/dense/Tensordot_1/Prod_1:output:0*
N*
T0*
_output_shapes
:�
8transformer_block/sequential/dense/Tensordot_1/transpose	Transpose;transformer_block/layer_normalization/batchnorm_1/add_1:z:0>transformer_block/sequential/dense/Tensordot_1/concat:output:0*
T0*,
_output_shapes
:���������-��
6transformer_block/sequential/dense/Tensordot_1/ReshapeReshape<transformer_block/sequential/dense/Tensordot_1/transpose:y:0=transformer_block/sequential/dense/Tensordot_1/stack:output:0*
T0*0
_output_shapes
:�������������������
5transformer_block/sequential/dense/Tensordot_1/MatMulMatMul?transformer_block/sequential/dense/Tensordot_1/Reshape:output:0Etransformer_block/sequential/dense/Tensordot_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
6transformer_block/sequential/dense/Tensordot_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�~
<transformer_block/sequential/dense/Tensordot_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot_1/concat_1ConcatV2@transformer_block/sequential/dense/Tensordot_1/GatherV2:output:0?transformer_block/sequential/dense/Tensordot_1/Const_2:output:0Etransformer_block/sequential/dense/Tensordot_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
.transformer_block/sequential/dense/Tensordot_1Reshape?transformer_block/sequential/dense/Tensordot_1/MatMul:product:0@transformer_block/sequential/dense/Tensordot_1/concat_1:output:0*
T0*,
_output_shapes
:���������-��
;transformer_block/sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOpBtransformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,transformer_block/sequential/dense/BiasAdd_1BiasAdd7transformer_block/sequential/dense/Tensordot_1:output:0Ctransformer_block/sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
)transformer_block/sequential/dense/Selu_1Selu5transformer_block/sequential/dense/BiasAdd_1:output:0*
T0*,
_output_shapes
:���������-��
?transformer_block/sequential/dense_1/Tensordot_1/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0
5transformer_block/sequential/dense_1/Tensordot_1/axesConst*
_output_shapes
:*
dtype0*
valueB:�
5transformer_block/sequential/dense_1/Tensordot_1/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
6transformer_block/sequential/dense_1/Tensordot_1/ShapeShape7transformer_block/sequential/dense/Selu_1:activations:0*
T0*
_output_shapes
:�
>transformer_block/sequential/dense_1/Tensordot_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot_1/GatherV2GatherV2?transformer_block/sequential/dense_1/Tensordot_1/Shape:output:0>transformer_block/sequential/dense_1/Tensordot_1/free:output:0Gtransformer_block/sequential/dense_1/Tensordot_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
@transformer_block/sequential/dense_1/Tensordot_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_block/sequential/dense_1/Tensordot_1/GatherV2_1GatherV2?transformer_block/sequential/dense_1/Tensordot_1/Shape:output:0>transformer_block/sequential/dense_1/Tensordot_1/axes:output:0Itransformer_block/sequential/dense_1/Tensordot_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
6transformer_block/sequential/dense_1/Tensordot_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense_1/Tensordot_1/ProdProdBtransformer_block/sequential/dense_1/Tensordot_1/GatherV2:output:0?transformer_block/sequential/dense_1/Tensordot_1/Const:output:0*
T0*
_output_shapes
: �
8transformer_block/sequential/dense_1/Tensordot_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
7transformer_block/sequential/dense_1/Tensordot_1/Prod_1ProdDtransformer_block/sequential/dense_1/Tensordot_1/GatherV2_1:output:0Atransformer_block/sequential/dense_1/Tensordot_1/Const_1:output:0*
T0*
_output_shapes
: ~
<transformer_block/sequential/dense_1/Tensordot_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense_1/Tensordot_1/concatConcatV2>transformer_block/sequential/dense_1/Tensordot_1/free:output:0>transformer_block/sequential/dense_1/Tensordot_1/axes:output:0Etransformer_block/sequential/dense_1/Tensordot_1/concat/axis:output:0*
N*
T0*
_output_shapes
:�
6transformer_block/sequential/dense_1/Tensordot_1/stackPack>transformer_block/sequential/dense_1/Tensordot_1/Prod:output:0@transformer_block/sequential/dense_1/Tensordot_1/Prod_1:output:0*
N*
T0*
_output_shapes
:�
:transformer_block/sequential/dense_1/Tensordot_1/transpose	Transpose7transformer_block/sequential/dense/Selu_1:activations:0@transformer_block/sequential/dense_1/Tensordot_1/concat:output:0*
T0*,
_output_shapes
:���������-��
8transformer_block/sequential/dense_1/Tensordot_1/ReshapeReshape>transformer_block/sequential/dense_1/Tensordot_1/transpose:y:0?transformer_block/sequential/dense_1/Tensordot_1/stack:output:0*
T0*0
_output_shapes
:�������������������
7transformer_block/sequential/dense_1/Tensordot_1/MatMulMatMulAtransformer_block/sequential/dense_1/Tensordot_1/Reshape:output:0Gtransformer_block/sequential/dense_1/Tensordot_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
8transformer_block/sequential/dense_1/Tensordot_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
>transformer_block/sequential/dense_1/Tensordot_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot_1/concat_1ConcatV2Btransformer_block/sequential/dense_1/Tensordot_1/GatherV2:output:0Atransformer_block/sequential/dense_1/Tensordot_1/Const_2:output:0Gtransformer_block/sequential/dense_1/Tensordot_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
0transformer_block/sequential/dense_1/Tensordot_1ReshapeAtransformer_block/sequential/dense_1/Tensordot_1/MatMul:product:0Btransformer_block/sequential/dense_1/Tensordot_1/concat_1:output:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.transformer_block/sequential/dense_1/BiasAdd_1BiasAdd9transformer_block/sequential/dense_1/Tensordot_1:output:0Etransformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�p
+transformer_block/dropout_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
)transformer_block/dropout_1/dropout_1/MulMul7transformer_block/sequential/dense_1/BiasAdd_1:output:04transformer_block/dropout_1/dropout_1/Const:output:0*
T0*,
_output_shapes
:���������-��
+transformer_block/dropout_1/dropout_1/ShapeShape7transformer_block/sequential/dense_1/BiasAdd_1:output:0*
T0*
_output_shapes
:�
Btransformer_block/dropout_1/dropout_1/random_uniform/RandomUniformRandomUniform4transformer_block/dropout_1/dropout_1/Shape:output:0*
T0*,
_output_shapes
:���������-�*
dtype0y
4transformer_block/dropout_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *U;>�
2transformer_block/dropout_1/dropout_1/GreaterEqualGreaterEqualKtransformer_block/dropout_1/dropout_1/random_uniform/RandomUniform:output:0=transformer_block/dropout_1/dropout_1/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������-��
*transformer_block/dropout_1/dropout_1/CastCast6transformer_block/dropout_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������-��
+transformer_block/dropout_1/dropout_1/Mul_1Mul-transformer_block/dropout_1/dropout_1/Mul:z:0.transformer_block/dropout_1/dropout_1/Cast:y:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_3AddV2;transformer_block/layer_normalization/batchnorm_1/add_1:z:0/transformer_block/dropout_1/dropout_1/Mul_1:z:0*
T0*,
_output_shapes
:���������-��
Htransformer_block/layer_normalization_1/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
6transformer_block/layer_normalization_1/moments_1/meanMeantransformer_block/add_3:z:0Qtransformer_block/layer_normalization_1/moments_1/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
>transformer_block/layer_normalization_1/moments_1/StopGradientStopGradient?transformer_block/layer_normalization_1/moments_1/mean:output:0*
T0*+
_output_shapes
:���������-�
Ctransformer_block/layer_normalization_1/moments_1/SquaredDifferenceSquaredDifferencetransformer_block/add_3:z:0Gtransformer_block/layer_normalization_1/moments_1/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Ltransformer_block/layer_normalization_1/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:transformer_block/layer_normalization_1/moments_1/varianceMeanGtransformer_block/layer_normalization_1/moments_1/SquaredDifference:z:0Utransformer_block/layer_normalization_1/moments_1/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(~
9transformer_block/layer_normalization_1/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
7transformer_block/layer_normalization_1/batchnorm_1/addAddV2Ctransformer_block/layer_normalization_1/moments_1/variance:output:0Btransformer_block/layer_normalization_1/batchnorm_1/add/y:output:0*
T0*+
_output_shapes
:���������-�
9transformer_block/layer_normalization_1/batchnorm_1/RsqrtRsqrt;transformer_block/layer_normalization_1/batchnorm_1/add:z:0*
T0*+
_output_shapes
:���������-�
Ftransformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_block/layer_normalization_1/batchnorm_1/mulMul=transformer_block/layer_normalization_1/batchnorm_1/Rsqrt:y:0Ntransformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_1/mul_1Multransformer_block/add_3:z:0;transformer_block/layer_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_1/mul_2Mul?transformer_block/layer_normalization_1/moments_1/mean:output:0;transformer_block/layer_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:���������-��
Btransformer_block/layer_normalization_1/batchnorm_1/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_block/layer_normalization_1/batchnorm_1/subSubJtransformer_block/layer_normalization_1/batchnorm_1/ReadVariableOp:value:0=transformer_block/layer_normalization_1/batchnorm_1/mul_2:z:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_1/add_1AddV2=transformer_block/layer_normalization_1/batchnorm_1/mul_1:z:0;transformer_block/layer_normalization_1/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:���������-��
Ktransformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_block/multi_head_attention/query/einsum_2/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_1/add_1:z:0Stransformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Atransformer_block/multi_head_attention/query/add_2/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_block/multi_head_attention/query/add_2AddV2Etransformer_block/multi_head_attention/query/einsum_2/Einsum:output:0Itransformer_block/multi_head_attention/query/add_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Itransformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOpReadVariableOpPtransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
:transformer_block/multi_head_attention/key/einsum_2/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_1/add_1:z:0Qtransformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
?transformer_block/multi_head_attention/key/add_2/ReadVariableOpReadVariableOpFtransformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
0transformer_block/multi_head_attention/key/add_2AddV2Ctransformer_block/multi_head_attention/key/einsum_2/Einsum:output:0Gtransformer_block/multi_head_attention/key/add_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Ktransformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_block/multi_head_attention/value/einsum_2/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_1/add_1:z:0Stransformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Atransformer_block/multi_head_attention/value/add_2/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_block/multi_head_attention/value/add_2AddV2Etransformer_block/multi_head_attention/value/einsum_2/Einsum:output:0Itransformer_block/multi_head_attention/value/add_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�s
.transformer_block/multi_head_attention/Mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
,transformer_block/multi_head_attention/Mul_2Mul6transformer_block/multi_head_attention/query/add_2:z:07transformer_block/multi_head_attention/Mul_2/y:output:0*
T0*0
_output_shapes
:���������-��
6transformer_block/multi_head_attention/einsum_4/EinsumEinsum4transformer_block/multi_head_attention/key/add_2:z:00transformer_block/multi_head_attention/Mul_2:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
8transformer_block/multi_head_attention/softmax/Softmax_2Softmax?transformer_block/multi_head_attention/einsum_4/Einsum:output:0*
T0*/
_output_shapes
:���������--�
6transformer_block/multi_head_attention/einsum_5/EinsumEinsumBtransformer_block/multi_head_attention/softmax/Softmax_2:softmax:06transformer_block/multi_head_attention/value/add_2:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
Vtransformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOpReadVariableOp]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Gtransformer_block/multi_head_attention/attention_output/einsum_2/EinsumEinsum?transformer_block/multi_head_attention/einsum_5/Einsum:output:0^transformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
Ltransformer_block/multi_head_attention/attention_output/add_2/ReadVariableOpReadVariableOpStransformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=transformer_block/multi_head_attention/attention_output/add_2AddV2Ptransformer_block/multi_head_attention/attention_output/einsum_2/Einsum:output:0Ttransformer_block/multi_head_attention/attention_output/add_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�n
)transformer_block/dropout/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
'transformer_block/dropout/dropout_2/MulMulAtransformer_block/multi_head_attention/attention_output/add_2:z:02transformer_block/dropout/dropout_2/Const:output:0*
T0*,
_output_shapes
:���������-��
)transformer_block/dropout/dropout_2/ShapeShapeAtransformer_block/multi_head_attention/attention_output/add_2:z:0*
T0*
_output_shapes
:�
@transformer_block/dropout/dropout_2/random_uniform/RandomUniformRandomUniform2transformer_block/dropout/dropout_2/Shape:output:0*
T0*,
_output_shapes
:���������-�*
dtype0w
2transformer_block/dropout/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *U;>�
0transformer_block/dropout/dropout_2/GreaterEqualGreaterEqualItransformer_block/dropout/dropout_2/random_uniform/RandomUniform:output:0;transformer_block/dropout/dropout_2/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������-��
(transformer_block/dropout/dropout_2/CastCast4transformer_block/dropout/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������-��
)transformer_block/dropout/dropout_2/Mul_1Mul+transformer_block/dropout/dropout_2/Mul:z:0,transformer_block/dropout/dropout_2/Cast:y:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_4AddV2=transformer_block/layer_normalization_1/batchnorm_1/add_1:z:0-transformer_block/dropout/dropout_2/Mul_1:z:0*
T0*,
_output_shapes
:���������-��
Ftransformer_block/layer_normalization/moments_2/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
4transformer_block/layer_normalization/moments_2/meanMeantransformer_block/add_4:z:0Otransformer_block/layer_normalization/moments_2/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
<transformer_block/layer_normalization/moments_2/StopGradientStopGradient=transformer_block/layer_normalization/moments_2/mean:output:0*
T0*+
_output_shapes
:���������-�
Atransformer_block/layer_normalization/moments_2/SquaredDifferenceSquaredDifferencetransformer_block/add_4:z:0Etransformer_block/layer_normalization/moments_2/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Jtransformer_block/layer_normalization/moments_2/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
8transformer_block/layer_normalization/moments_2/varianceMeanEtransformer_block/layer_normalization/moments_2/SquaredDifference:z:0Stransformer_block/layer_normalization/moments_2/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(|
7transformer_block/layer_normalization/batchnorm_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
5transformer_block/layer_normalization/batchnorm_2/addAddV2Atransformer_block/layer_normalization/moments_2/variance:output:0@transformer_block/layer_normalization/batchnorm_2/add/y:output:0*
T0*+
_output_shapes
:���������-�
7transformer_block/layer_normalization/batchnorm_2/RsqrtRsqrt9transformer_block/layer_normalization/batchnorm_2/add:z:0*
T0*+
_output_shapes
:���������-�
Dtransformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization/batchnorm_2/mulMul;transformer_block/layer_normalization/batchnorm_2/Rsqrt:y:0Ltransformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_2/mul_1Multransformer_block/add_4:z:09transformer_block/layer_normalization/batchnorm_2/mul:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_2/mul_2Mul=transformer_block/layer_normalization/moments_2/mean:output:09transformer_block/layer_normalization/batchnorm_2/mul:z:0*
T0*,
_output_shapes
:���������-��
@transformer_block/layer_normalization/batchnorm_2/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization/batchnorm_2/subSubHtransformer_block/layer_normalization/batchnorm_2/ReadVariableOp:value:0;transformer_block/layer_normalization/batchnorm_2/mul_2:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_2/add_1AddV2;transformer_block/layer_normalization/batchnorm_2/mul_1:z:09transformer_block/layer_normalization/batchnorm_2/sub:z:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense/Tensordot_2/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0}
3transformer_block/sequential/dense/Tensordot_2/axesConst*
_output_shapes
:*
dtype0*
valueB:�
3transformer_block/sequential/dense/Tensordot_2/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
4transformer_block/sequential/dense/Tensordot_2/ShapeShape;transformer_block/layer_normalization/batchnorm_2/add_1:z:0*
T0*
_output_shapes
:~
<transformer_block/sequential/dense/Tensordot_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot_2/GatherV2GatherV2=transformer_block/sequential/dense/Tensordot_2/Shape:output:0<transformer_block/sequential/dense/Tensordot_2/free:output:0Etransformer_block/sequential/dense/Tensordot_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
>transformer_block/sequential/dense/Tensordot_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense/Tensordot_2/GatherV2_1GatherV2=transformer_block/sequential/dense/Tensordot_2/Shape:output:0<transformer_block/sequential/dense/Tensordot_2/axes:output:0Gtransformer_block/sequential/dense/Tensordot_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4transformer_block/sequential/dense/Tensordot_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
3transformer_block/sequential/dense/Tensordot_2/ProdProd@transformer_block/sequential/dense/Tensordot_2/GatherV2:output:0=transformer_block/sequential/dense/Tensordot_2/Const:output:0*
T0*
_output_shapes
: �
6transformer_block/sequential/dense/Tensordot_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense/Tensordot_2/Prod_1ProdBtransformer_block/sequential/dense/Tensordot_2/GatherV2_1:output:0?transformer_block/sequential/dense/Tensordot_2/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_block/sequential/dense/Tensordot_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5transformer_block/sequential/dense/Tensordot_2/concatConcatV2<transformer_block/sequential/dense/Tensordot_2/free:output:0<transformer_block/sequential/dense/Tensordot_2/axes:output:0Ctransformer_block/sequential/dense/Tensordot_2/concat/axis:output:0*
N*
T0*
_output_shapes
:�
4transformer_block/sequential/dense/Tensordot_2/stackPack<transformer_block/sequential/dense/Tensordot_2/Prod:output:0>transformer_block/sequential/dense/Tensordot_2/Prod_1:output:0*
N*
T0*
_output_shapes
:�
8transformer_block/sequential/dense/Tensordot_2/transpose	Transpose;transformer_block/layer_normalization/batchnorm_2/add_1:z:0>transformer_block/sequential/dense/Tensordot_2/concat:output:0*
T0*,
_output_shapes
:���������-��
6transformer_block/sequential/dense/Tensordot_2/ReshapeReshape<transformer_block/sequential/dense/Tensordot_2/transpose:y:0=transformer_block/sequential/dense/Tensordot_2/stack:output:0*
T0*0
_output_shapes
:�������������������
5transformer_block/sequential/dense/Tensordot_2/MatMulMatMul?transformer_block/sequential/dense/Tensordot_2/Reshape:output:0Etransformer_block/sequential/dense/Tensordot_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
6transformer_block/sequential/dense/Tensordot_2/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�~
<transformer_block/sequential/dense/Tensordot_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot_2/concat_1ConcatV2@transformer_block/sequential/dense/Tensordot_2/GatherV2:output:0?transformer_block/sequential/dense/Tensordot_2/Const_2:output:0Etransformer_block/sequential/dense/Tensordot_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
.transformer_block/sequential/dense/Tensordot_2Reshape?transformer_block/sequential/dense/Tensordot_2/MatMul:product:0@transformer_block/sequential/dense/Tensordot_2/concat_1:output:0*
T0*,
_output_shapes
:���������-��
;transformer_block/sequential/dense/BiasAdd_2/ReadVariableOpReadVariableOpBtransformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,transformer_block/sequential/dense/BiasAdd_2BiasAdd7transformer_block/sequential/dense/Tensordot_2:output:0Ctransformer_block/sequential/dense/BiasAdd_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
)transformer_block/sequential/dense/Selu_2Selu5transformer_block/sequential/dense/BiasAdd_2:output:0*
T0*,
_output_shapes
:���������-��
?transformer_block/sequential/dense_1/Tensordot_2/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0
5transformer_block/sequential/dense_1/Tensordot_2/axesConst*
_output_shapes
:*
dtype0*
valueB:�
5transformer_block/sequential/dense_1/Tensordot_2/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
6transformer_block/sequential/dense_1/Tensordot_2/ShapeShape7transformer_block/sequential/dense/Selu_2:activations:0*
T0*
_output_shapes
:�
>transformer_block/sequential/dense_1/Tensordot_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot_2/GatherV2GatherV2?transformer_block/sequential/dense_1/Tensordot_2/Shape:output:0>transformer_block/sequential/dense_1/Tensordot_2/free:output:0Gtransformer_block/sequential/dense_1/Tensordot_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
@transformer_block/sequential/dense_1/Tensordot_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_block/sequential/dense_1/Tensordot_2/GatherV2_1GatherV2?transformer_block/sequential/dense_1/Tensordot_2/Shape:output:0>transformer_block/sequential/dense_1/Tensordot_2/axes:output:0Itransformer_block/sequential/dense_1/Tensordot_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
6transformer_block/sequential/dense_1/Tensordot_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense_1/Tensordot_2/ProdProdBtransformer_block/sequential/dense_1/Tensordot_2/GatherV2:output:0?transformer_block/sequential/dense_1/Tensordot_2/Const:output:0*
T0*
_output_shapes
: �
8transformer_block/sequential/dense_1/Tensordot_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
7transformer_block/sequential/dense_1/Tensordot_2/Prod_1ProdDtransformer_block/sequential/dense_1/Tensordot_2/GatherV2_1:output:0Atransformer_block/sequential/dense_1/Tensordot_2/Const_1:output:0*
T0*
_output_shapes
: ~
<transformer_block/sequential/dense_1/Tensordot_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense_1/Tensordot_2/concatConcatV2>transformer_block/sequential/dense_1/Tensordot_2/free:output:0>transformer_block/sequential/dense_1/Tensordot_2/axes:output:0Etransformer_block/sequential/dense_1/Tensordot_2/concat/axis:output:0*
N*
T0*
_output_shapes
:�
6transformer_block/sequential/dense_1/Tensordot_2/stackPack>transformer_block/sequential/dense_1/Tensordot_2/Prod:output:0@transformer_block/sequential/dense_1/Tensordot_2/Prod_1:output:0*
N*
T0*
_output_shapes
:�
:transformer_block/sequential/dense_1/Tensordot_2/transpose	Transpose7transformer_block/sequential/dense/Selu_2:activations:0@transformer_block/sequential/dense_1/Tensordot_2/concat:output:0*
T0*,
_output_shapes
:���������-��
8transformer_block/sequential/dense_1/Tensordot_2/ReshapeReshape>transformer_block/sequential/dense_1/Tensordot_2/transpose:y:0?transformer_block/sequential/dense_1/Tensordot_2/stack:output:0*
T0*0
_output_shapes
:�������������������
7transformer_block/sequential/dense_1/Tensordot_2/MatMulMatMulAtransformer_block/sequential/dense_1/Tensordot_2/Reshape:output:0Gtransformer_block/sequential/dense_1/Tensordot_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
8transformer_block/sequential/dense_1/Tensordot_2/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
>transformer_block/sequential/dense_1/Tensordot_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot_2/concat_1ConcatV2Btransformer_block/sequential/dense_1/Tensordot_2/GatherV2:output:0Atransformer_block/sequential/dense_1/Tensordot_2/Const_2:output:0Gtransformer_block/sequential/dense_1/Tensordot_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
0transformer_block/sequential/dense_1/Tensordot_2ReshapeAtransformer_block/sequential/dense_1/Tensordot_2/MatMul:product:0Btransformer_block/sequential/dense_1/Tensordot_2/concat_1:output:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.transformer_block/sequential/dense_1/BiasAdd_2BiasAdd9transformer_block/sequential/dense_1/Tensordot_2:output:0Etransformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�p
+transformer_block/dropout_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
)transformer_block/dropout_1/dropout_2/MulMul7transformer_block/sequential/dense_1/BiasAdd_2:output:04transformer_block/dropout_1/dropout_2/Const:output:0*
T0*,
_output_shapes
:���������-��
+transformer_block/dropout_1/dropout_2/ShapeShape7transformer_block/sequential/dense_1/BiasAdd_2:output:0*
T0*
_output_shapes
:�
Btransformer_block/dropout_1/dropout_2/random_uniform/RandomUniformRandomUniform4transformer_block/dropout_1/dropout_2/Shape:output:0*
T0*,
_output_shapes
:���������-�*
dtype0y
4transformer_block/dropout_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *U;>�
2transformer_block/dropout_1/dropout_2/GreaterEqualGreaterEqualKtransformer_block/dropout_1/dropout_2/random_uniform/RandomUniform:output:0=transformer_block/dropout_1/dropout_2/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������-��
*transformer_block/dropout_1/dropout_2/CastCast6transformer_block/dropout_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������-��
+transformer_block/dropout_1/dropout_2/Mul_1Mul-transformer_block/dropout_1/dropout_2/Mul:z:0.transformer_block/dropout_1/dropout_2/Cast:y:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_5AddV2;transformer_block/layer_normalization/batchnorm_2/add_1:z:0/transformer_block/dropout_1/dropout_2/Mul_1:z:0*
T0*,
_output_shapes
:���������-��
Htransformer_block/layer_normalization_1/moments_2/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
6transformer_block/layer_normalization_1/moments_2/meanMeantransformer_block/add_5:z:0Qtransformer_block/layer_normalization_1/moments_2/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
>transformer_block/layer_normalization_1/moments_2/StopGradientStopGradient?transformer_block/layer_normalization_1/moments_2/mean:output:0*
T0*+
_output_shapes
:���������-�
Ctransformer_block/layer_normalization_1/moments_2/SquaredDifferenceSquaredDifferencetransformer_block/add_5:z:0Gtransformer_block/layer_normalization_1/moments_2/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Ltransformer_block/layer_normalization_1/moments_2/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:transformer_block/layer_normalization_1/moments_2/varianceMeanGtransformer_block/layer_normalization_1/moments_2/SquaredDifference:z:0Utransformer_block/layer_normalization_1/moments_2/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(~
9transformer_block/layer_normalization_1/batchnorm_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
7transformer_block/layer_normalization_1/batchnorm_2/addAddV2Ctransformer_block/layer_normalization_1/moments_2/variance:output:0Btransformer_block/layer_normalization_1/batchnorm_2/add/y:output:0*
T0*+
_output_shapes
:���������-�
9transformer_block/layer_normalization_1/batchnorm_2/RsqrtRsqrt;transformer_block/layer_normalization_1/batchnorm_2/add:z:0*
T0*+
_output_shapes
:���������-�
Ftransformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_block/layer_normalization_1/batchnorm_2/mulMul=transformer_block/layer_normalization_1/batchnorm_2/Rsqrt:y:0Ntransformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_2/mul_1Multransformer_block/add_5:z:0;transformer_block/layer_normalization_1/batchnorm_2/mul:z:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_2/mul_2Mul?transformer_block/layer_normalization_1/moments_2/mean:output:0;transformer_block/layer_normalization_1/batchnorm_2/mul:z:0*
T0*,
_output_shapes
:���������-��
Btransformer_block/layer_normalization_1/batchnorm_2/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_block/layer_normalization_1/batchnorm_2/subSubJtransformer_block/layer_normalization_1/batchnorm_2/ReadVariableOp:value:0=transformer_block/layer_normalization_1/batchnorm_2/mul_2:z:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_2/add_1AddV2=transformer_block/layer_normalization_1/batchnorm_2/mul_1:z:0;transformer_block/layer_normalization_1/batchnorm_2/sub:z:0*
T0*,
_output_shapes
:���������-��
Ktransformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_block/multi_head_attention/query/einsum_3/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_2/add_1:z:0Stransformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Atransformer_block/multi_head_attention/query/add_3/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_block/multi_head_attention/query/add_3AddV2Etransformer_block/multi_head_attention/query/einsum_3/Einsum:output:0Itransformer_block/multi_head_attention/query/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Itransformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOpReadVariableOpPtransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
:transformer_block/multi_head_attention/key/einsum_3/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_2/add_1:z:0Qtransformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
?transformer_block/multi_head_attention/key/add_3/ReadVariableOpReadVariableOpFtransformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
0transformer_block/multi_head_attention/key/add_3AddV2Ctransformer_block/multi_head_attention/key/einsum_3/Einsum:output:0Gtransformer_block/multi_head_attention/key/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Ktransformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_block/multi_head_attention/value/einsum_3/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_2/add_1:z:0Stransformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Atransformer_block/multi_head_attention/value/add_3/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_block/multi_head_attention/value/add_3AddV2Etransformer_block/multi_head_attention/value/einsum_3/Einsum:output:0Itransformer_block/multi_head_attention/value/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�s
.transformer_block/multi_head_attention/Mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
,transformer_block/multi_head_attention/Mul_3Mul6transformer_block/multi_head_attention/query/add_3:z:07transformer_block/multi_head_attention/Mul_3/y:output:0*
T0*0
_output_shapes
:���������-��
6transformer_block/multi_head_attention/einsum_6/EinsumEinsum4transformer_block/multi_head_attention/key/add_3:z:00transformer_block/multi_head_attention/Mul_3:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
8transformer_block/multi_head_attention/softmax/Softmax_3Softmax?transformer_block/multi_head_attention/einsum_6/Einsum:output:0*
T0*/
_output_shapes
:���������--�
6transformer_block/multi_head_attention/einsum_7/EinsumEinsumBtransformer_block/multi_head_attention/softmax/Softmax_3:softmax:06transformer_block/multi_head_attention/value/add_3:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
Vtransformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOpReadVariableOp]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Gtransformer_block/multi_head_attention/attention_output/einsum_3/EinsumEinsum?transformer_block/multi_head_attention/einsum_7/Einsum:output:0^transformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
Ltransformer_block/multi_head_attention/attention_output/add_3/ReadVariableOpReadVariableOpStransformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=transformer_block/multi_head_attention/attention_output/add_3AddV2Ptransformer_block/multi_head_attention/attention_output/einsum_3/Einsum:output:0Ttransformer_block/multi_head_attention/attention_output/add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�n
)transformer_block/dropout/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
'transformer_block/dropout/dropout_3/MulMulAtransformer_block/multi_head_attention/attention_output/add_3:z:02transformer_block/dropout/dropout_3/Const:output:0*
T0*,
_output_shapes
:���������-��
)transformer_block/dropout/dropout_3/ShapeShapeAtransformer_block/multi_head_attention/attention_output/add_3:z:0*
T0*
_output_shapes
:�
@transformer_block/dropout/dropout_3/random_uniform/RandomUniformRandomUniform2transformer_block/dropout/dropout_3/Shape:output:0*
T0*,
_output_shapes
:���������-�*
dtype0w
2transformer_block/dropout/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *U;>�
0transformer_block/dropout/dropout_3/GreaterEqualGreaterEqualItransformer_block/dropout/dropout_3/random_uniform/RandomUniform:output:0;transformer_block/dropout/dropout_3/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������-��
(transformer_block/dropout/dropout_3/CastCast4transformer_block/dropout/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������-��
)transformer_block/dropout/dropout_3/Mul_1Mul+transformer_block/dropout/dropout_3/Mul:z:0,transformer_block/dropout/dropout_3/Cast:y:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_6AddV2=transformer_block/layer_normalization_1/batchnorm_2/add_1:z:0-transformer_block/dropout/dropout_3/Mul_1:z:0*
T0*,
_output_shapes
:���������-��
Ftransformer_block/layer_normalization/moments_3/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
4transformer_block/layer_normalization/moments_3/meanMeantransformer_block/add_6:z:0Otransformer_block/layer_normalization/moments_3/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
<transformer_block/layer_normalization/moments_3/StopGradientStopGradient=transformer_block/layer_normalization/moments_3/mean:output:0*
T0*+
_output_shapes
:���������-�
Atransformer_block/layer_normalization/moments_3/SquaredDifferenceSquaredDifferencetransformer_block/add_6:z:0Etransformer_block/layer_normalization/moments_3/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Jtransformer_block/layer_normalization/moments_3/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
8transformer_block/layer_normalization/moments_3/varianceMeanEtransformer_block/layer_normalization/moments_3/SquaredDifference:z:0Stransformer_block/layer_normalization/moments_3/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(|
7transformer_block/layer_normalization/batchnorm_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
5transformer_block/layer_normalization/batchnorm_3/addAddV2Atransformer_block/layer_normalization/moments_3/variance:output:0@transformer_block/layer_normalization/batchnorm_3/add/y:output:0*
T0*+
_output_shapes
:���������-�
7transformer_block/layer_normalization/batchnorm_3/RsqrtRsqrt9transformer_block/layer_normalization/batchnorm_3/add:z:0*
T0*+
_output_shapes
:���������-�
Dtransformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization/batchnorm_3/mulMul;transformer_block/layer_normalization/batchnorm_3/Rsqrt:y:0Ltransformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_3/mul_1Multransformer_block/add_6:z:09transformer_block/layer_normalization/batchnorm_3/mul:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_3/mul_2Mul=transformer_block/layer_normalization/moments_3/mean:output:09transformer_block/layer_normalization/batchnorm_3/mul:z:0*
T0*,
_output_shapes
:���������-��
@transformer_block/layer_normalization/batchnorm_3/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization/batchnorm_3/subSubHtransformer_block/layer_normalization/batchnorm_3/ReadVariableOp:value:0;transformer_block/layer_normalization/batchnorm_3/mul_2:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_3/add_1AddV2;transformer_block/layer_normalization/batchnorm_3/mul_1:z:09transformer_block/layer_normalization/batchnorm_3/sub:z:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense/Tensordot_3/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0}
3transformer_block/sequential/dense/Tensordot_3/axesConst*
_output_shapes
:*
dtype0*
valueB:�
3transformer_block/sequential/dense/Tensordot_3/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
4transformer_block/sequential/dense/Tensordot_3/ShapeShape;transformer_block/layer_normalization/batchnorm_3/add_1:z:0*
T0*
_output_shapes
:~
<transformer_block/sequential/dense/Tensordot_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot_3/GatherV2GatherV2=transformer_block/sequential/dense/Tensordot_3/Shape:output:0<transformer_block/sequential/dense/Tensordot_3/free:output:0Etransformer_block/sequential/dense/Tensordot_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
>transformer_block/sequential/dense/Tensordot_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense/Tensordot_3/GatherV2_1GatherV2=transformer_block/sequential/dense/Tensordot_3/Shape:output:0<transformer_block/sequential/dense/Tensordot_3/axes:output:0Gtransformer_block/sequential/dense/Tensordot_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4transformer_block/sequential/dense/Tensordot_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
3transformer_block/sequential/dense/Tensordot_3/ProdProd@transformer_block/sequential/dense/Tensordot_3/GatherV2:output:0=transformer_block/sequential/dense/Tensordot_3/Const:output:0*
T0*
_output_shapes
: �
6transformer_block/sequential/dense/Tensordot_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense/Tensordot_3/Prod_1ProdBtransformer_block/sequential/dense/Tensordot_3/GatherV2_1:output:0?transformer_block/sequential/dense/Tensordot_3/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_block/sequential/dense/Tensordot_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5transformer_block/sequential/dense/Tensordot_3/concatConcatV2<transformer_block/sequential/dense/Tensordot_3/free:output:0<transformer_block/sequential/dense/Tensordot_3/axes:output:0Ctransformer_block/sequential/dense/Tensordot_3/concat/axis:output:0*
N*
T0*
_output_shapes
:�
4transformer_block/sequential/dense/Tensordot_3/stackPack<transformer_block/sequential/dense/Tensordot_3/Prod:output:0>transformer_block/sequential/dense/Tensordot_3/Prod_1:output:0*
N*
T0*
_output_shapes
:�
8transformer_block/sequential/dense/Tensordot_3/transpose	Transpose;transformer_block/layer_normalization/batchnorm_3/add_1:z:0>transformer_block/sequential/dense/Tensordot_3/concat:output:0*
T0*,
_output_shapes
:���������-��
6transformer_block/sequential/dense/Tensordot_3/ReshapeReshape<transformer_block/sequential/dense/Tensordot_3/transpose:y:0=transformer_block/sequential/dense/Tensordot_3/stack:output:0*
T0*0
_output_shapes
:�������������������
5transformer_block/sequential/dense/Tensordot_3/MatMulMatMul?transformer_block/sequential/dense/Tensordot_3/Reshape:output:0Etransformer_block/sequential/dense/Tensordot_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
6transformer_block/sequential/dense/Tensordot_3/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�~
<transformer_block/sequential/dense/Tensordot_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot_3/concat_1ConcatV2@transformer_block/sequential/dense/Tensordot_3/GatherV2:output:0?transformer_block/sequential/dense/Tensordot_3/Const_2:output:0Etransformer_block/sequential/dense/Tensordot_3/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
.transformer_block/sequential/dense/Tensordot_3Reshape?transformer_block/sequential/dense/Tensordot_3/MatMul:product:0@transformer_block/sequential/dense/Tensordot_3/concat_1:output:0*
T0*,
_output_shapes
:���������-��
;transformer_block/sequential/dense/BiasAdd_3/ReadVariableOpReadVariableOpBtransformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,transformer_block/sequential/dense/BiasAdd_3BiasAdd7transformer_block/sequential/dense/Tensordot_3:output:0Ctransformer_block/sequential/dense/BiasAdd_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
)transformer_block/sequential/dense/Selu_3Selu5transformer_block/sequential/dense/BiasAdd_3:output:0*
T0*,
_output_shapes
:���������-��
?transformer_block/sequential/dense_1/Tensordot_3/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0
5transformer_block/sequential/dense_1/Tensordot_3/axesConst*
_output_shapes
:*
dtype0*
valueB:�
5transformer_block/sequential/dense_1/Tensordot_3/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
6transformer_block/sequential/dense_1/Tensordot_3/ShapeShape7transformer_block/sequential/dense/Selu_3:activations:0*
T0*
_output_shapes
:�
>transformer_block/sequential/dense_1/Tensordot_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot_3/GatherV2GatherV2?transformer_block/sequential/dense_1/Tensordot_3/Shape:output:0>transformer_block/sequential/dense_1/Tensordot_3/free:output:0Gtransformer_block/sequential/dense_1/Tensordot_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
@transformer_block/sequential/dense_1/Tensordot_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_block/sequential/dense_1/Tensordot_3/GatherV2_1GatherV2?transformer_block/sequential/dense_1/Tensordot_3/Shape:output:0>transformer_block/sequential/dense_1/Tensordot_3/axes:output:0Itransformer_block/sequential/dense_1/Tensordot_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
6transformer_block/sequential/dense_1/Tensordot_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense_1/Tensordot_3/ProdProdBtransformer_block/sequential/dense_1/Tensordot_3/GatherV2:output:0?transformer_block/sequential/dense_1/Tensordot_3/Const:output:0*
T0*
_output_shapes
: �
8transformer_block/sequential/dense_1/Tensordot_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
7transformer_block/sequential/dense_1/Tensordot_3/Prod_1ProdDtransformer_block/sequential/dense_1/Tensordot_3/GatherV2_1:output:0Atransformer_block/sequential/dense_1/Tensordot_3/Const_1:output:0*
T0*
_output_shapes
: ~
<transformer_block/sequential/dense_1/Tensordot_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense_1/Tensordot_3/concatConcatV2>transformer_block/sequential/dense_1/Tensordot_3/free:output:0>transformer_block/sequential/dense_1/Tensordot_3/axes:output:0Etransformer_block/sequential/dense_1/Tensordot_3/concat/axis:output:0*
N*
T0*
_output_shapes
:�
6transformer_block/sequential/dense_1/Tensordot_3/stackPack>transformer_block/sequential/dense_1/Tensordot_3/Prod:output:0@transformer_block/sequential/dense_1/Tensordot_3/Prod_1:output:0*
N*
T0*
_output_shapes
:�
:transformer_block/sequential/dense_1/Tensordot_3/transpose	Transpose7transformer_block/sequential/dense/Selu_3:activations:0@transformer_block/sequential/dense_1/Tensordot_3/concat:output:0*
T0*,
_output_shapes
:���������-��
8transformer_block/sequential/dense_1/Tensordot_3/ReshapeReshape>transformer_block/sequential/dense_1/Tensordot_3/transpose:y:0?transformer_block/sequential/dense_1/Tensordot_3/stack:output:0*
T0*0
_output_shapes
:�������������������
7transformer_block/sequential/dense_1/Tensordot_3/MatMulMatMulAtransformer_block/sequential/dense_1/Tensordot_3/Reshape:output:0Gtransformer_block/sequential/dense_1/Tensordot_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
8transformer_block/sequential/dense_1/Tensordot_3/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
>transformer_block/sequential/dense_1/Tensordot_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot_3/concat_1ConcatV2Btransformer_block/sequential/dense_1/Tensordot_3/GatherV2:output:0Atransformer_block/sequential/dense_1/Tensordot_3/Const_2:output:0Gtransformer_block/sequential/dense_1/Tensordot_3/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
0transformer_block/sequential/dense_1/Tensordot_3ReshapeAtransformer_block/sequential/dense_1/Tensordot_3/MatMul:product:0Btransformer_block/sequential/dense_1/Tensordot_3/concat_1:output:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.transformer_block/sequential/dense_1/BiasAdd_3BiasAdd9transformer_block/sequential/dense_1/Tensordot_3:output:0Etransformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�p
+transformer_block/dropout_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
)transformer_block/dropout_1/dropout_3/MulMul7transformer_block/sequential/dense_1/BiasAdd_3:output:04transformer_block/dropout_1/dropout_3/Const:output:0*
T0*,
_output_shapes
:���������-��
+transformer_block/dropout_1/dropout_3/ShapeShape7transformer_block/sequential/dense_1/BiasAdd_3:output:0*
T0*
_output_shapes
:�
Btransformer_block/dropout_1/dropout_3/random_uniform/RandomUniformRandomUniform4transformer_block/dropout_1/dropout_3/Shape:output:0*
T0*,
_output_shapes
:���������-�*
dtype0y
4transformer_block/dropout_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *U;>�
2transformer_block/dropout_1/dropout_3/GreaterEqualGreaterEqualKtransformer_block/dropout_1/dropout_3/random_uniform/RandomUniform:output:0=transformer_block/dropout_1/dropout_3/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������-��
*transformer_block/dropout_1/dropout_3/CastCast6transformer_block/dropout_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������-��
+transformer_block/dropout_1/dropout_3/Mul_1Mul-transformer_block/dropout_1/dropout_3/Mul:z:0.transformer_block/dropout_1/dropout_3/Cast:y:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_7AddV2;transformer_block/layer_normalization/batchnorm_3/add_1:z:0/transformer_block/dropout_1/dropout_3/Mul_1:z:0*
T0*,
_output_shapes
:���������-��
Htransformer_block/layer_normalization_1/moments_3/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
6transformer_block/layer_normalization_1/moments_3/meanMeantransformer_block/add_7:z:0Qtransformer_block/layer_normalization_1/moments_3/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
>transformer_block/layer_normalization_1/moments_3/StopGradientStopGradient?transformer_block/layer_normalization_1/moments_3/mean:output:0*
T0*+
_output_shapes
:���������-�
Ctransformer_block/layer_normalization_1/moments_3/SquaredDifferenceSquaredDifferencetransformer_block/add_7:z:0Gtransformer_block/layer_normalization_1/moments_3/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Ltransformer_block/layer_normalization_1/moments_3/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:transformer_block/layer_normalization_1/moments_3/varianceMeanGtransformer_block/layer_normalization_1/moments_3/SquaredDifference:z:0Utransformer_block/layer_normalization_1/moments_3/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(~
9transformer_block/layer_normalization_1/batchnorm_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
7transformer_block/layer_normalization_1/batchnorm_3/addAddV2Ctransformer_block/layer_normalization_1/moments_3/variance:output:0Btransformer_block/layer_normalization_1/batchnorm_3/add/y:output:0*
T0*+
_output_shapes
:���������-�
9transformer_block/layer_normalization_1/batchnorm_3/RsqrtRsqrt;transformer_block/layer_normalization_1/batchnorm_3/add:z:0*
T0*+
_output_shapes
:���������-�
Ftransformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_block/layer_normalization_1/batchnorm_3/mulMul=transformer_block/layer_normalization_1/batchnorm_3/Rsqrt:y:0Ntransformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_3/mul_1Multransformer_block/add_7:z:0;transformer_block/layer_normalization_1/batchnorm_3/mul:z:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_3/mul_2Mul?transformer_block/layer_normalization_1/moments_3/mean:output:0;transformer_block/layer_normalization_1/batchnorm_3/mul:z:0*
T0*,
_output_shapes
:���������-��
Btransformer_block/layer_normalization_1/batchnorm_3/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_block/layer_normalization_1/batchnorm_3/subSubJtransformer_block/layer_normalization_1/batchnorm_3/ReadVariableOp:value:0=transformer_block/layer_normalization_1/batchnorm_3/mul_2:z:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_3/add_1AddV2=transformer_block/layer_normalization_1/batchnorm_3/mul_1:z:0;transformer_block/layer_normalization_1/batchnorm_3/sub:z:0*
T0*,
_output_shapes
:���������-��
Ktransformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_block/multi_head_attention/query/einsum_4/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_3/add_1:z:0Stransformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Atransformer_block/multi_head_attention/query/add_4/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_block/multi_head_attention/query/add_4AddV2Etransformer_block/multi_head_attention/query/einsum_4/Einsum:output:0Itransformer_block/multi_head_attention/query/add_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Itransformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOpReadVariableOpPtransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
:transformer_block/multi_head_attention/key/einsum_4/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_3/add_1:z:0Qtransformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
?transformer_block/multi_head_attention/key/add_4/ReadVariableOpReadVariableOpFtransformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
0transformer_block/multi_head_attention/key/add_4AddV2Ctransformer_block/multi_head_attention/key/einsum_4/Einsum:output:0Gtransformer_block/multi_head_attention/key/add_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Ktransformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
<transformer_block/multi_head_attention/value/einsum_4/EinsumEinsum=transformer_block/layer_normalization_1/batchnorm_3/add_1:z:0Stransformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Atransformer_block/multi_head_attention/value/add_4/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2transformer_block/multi_head_attention/value/add_4AddV2Etransformer_block/multi_head_attention/value/einsum_4/Einsum:output:0Itransformer_block/multi_head_attention/value/add_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�s
.transformer_block/multi_head_attention/Mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
,transformer_block/multi_head_attention/Mul_4Mul6transformer_block/multi_head_attention/query/add_4:z:07transformer_block/multi_head_attention/Mul_4/y:output:0*
T0*0
_output_shapes
:���������-��
6transformer_block/multi_head_attention/einsum_8/EinsumEinsum4transformer_block/multi_head_attention/key/add_4:z:00transformer_block/multi_head_attention/Mul_4:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
8transformer_block/multi_head_attention/softmax/Softmax_4Softmax?transformer_block/multi_head_attention/einsum_8/Einsum:output:0*
T0*/
_output_shapes
:���������--�
6transformer_block/multi_head_attention/einsum_9/EinsumEinsumBtransformer_block/multi_head_attention/softmax/Softmax_4:softmax:06transformer_block/multi_head_attention/value/add_4:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
Vtransformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOpReadVariableOp]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Gtransformer_block/multi_head_attention/attention_output/einsum_4/EinsumEinsum?transformer_block/multi_head_attention/einsum_9/Einsum:output:0^transformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
Ltransformer_block/multi_head_attention/attention_output/add_4/ReadVariableOpReadVariableOpStransformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=transformer_block/multi_head_attention/attention_output/add_4AddV2Ptransformer_block/multi_head_attention/attention_output/einsum_4/Einsum:output:0Ttransformer_block/multi_head_attention/attention_output/add_4/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�n
)transformer_block/dropout/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
'transformer_block/dropout/dropout_4/MulMulAtransformer_block/multi_head_attention/attention_output/add_4:z:02transformer_block/dropout/dropout_4/Const:output:0*
T0*,
_output_shapes
:���������-��
)transformer_block/dropout/dropout_4/ShapeShapeAtransformer_block/multi_head_attention/attention_output/add_4:z:0*
T0*
_output_shapes
:�
@transformer_block/dropout/dropout_4/random_uniform/RandomUniformRandomUniform2transformer_block/dropout/dropout_4/Shape:output:0*
T0*,
_output_shapes
:���������-�*
dtype0w
2transformer_block/dropout/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *U;>�
0transformer_block/dropout/dropout_4/GreaterEqualGreaterEqualItransformer_block/dropout/dropout_4/random_uniform/RandomUniform:output:0;transformer_block/dropout/dropout_4/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������-��
(transformer_block/dropout/dropout_4/CastCast4transformer_block/dropout/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������-��
)transformer_block/dropout/dropout_4/Mul_1Mul+transformer_block/dropout/dropout_4/Mul:z:0,transformer_block/dropout/dropout_4/Cast:y:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_8AddV2=transformer_block/layer_normalization_1/batchnorm_3/add_1:z:0-transformer_block/dropout/dropout_4/Mul_1:z:0*
T0*,
_output_shapes
:���������-��
Ftransformer_block/layer_normalization/moments_4/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
4transformer_block/layer_normalization/moments_4/meanMeantransformer_block/add_8:z:0Otransformer_block/layer_normalization/moments_4/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
<transformer_block/layer_normalization/moments_4/StopGradientStopGradient=transformer_block/layer_normalization/moments_4/mean:output:0*
T0*+
_output_shapes
:���������-�
Atransformer_block/layer_normalization/moments_4/SquaredDifferenceSquaredDifferencetransformer_block/add_8:z:0Etransformer_block/layer_normalization/moments_4/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Jtransformer_block/layer_normalization/moments_4/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
8transformer_block/layer_normalization/moments_4/varianceMeanEtransformer_block/layer_normalization/moments_4/SquaredDifference:z:0Stransformer_block/layer_normalization/moments_4/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(|
7transformer_block/layer_normalization/batchnorm_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
5transformer_block/layer_normalization/batchnorm_4/addAddV2Atransformer_block/layer_normalization/moments_4/variance:output:0@transformer_block/layer_normalization/batchnorm_4/add/y:output:0*
T0*+
_output_shapes
:���������-�
7transformer_block/layer_normalization/batchnorm_4/RsqrtRsqrt9transformer_block/layer_normalization/batchnorm_4/add:z:0*
T0*+
_output_shapes
:���������-�
Dtransformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization/batchnorm_4/mulMul;transformer_block/layer_normalization/batchnorm_4/Rsqrt:y:0Ltransformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_4/mul_1Multransformer_block/add_8:z:09transformer_block/layer_normalization/batchnorm_4/mul:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_4/mul_2Mul=transformer_block/layer_normalization/moments_4/mean:output:09transformer_block/layer_normalization/batchnorm_4/mul:z:0*
T0*,
_output_shapes
:���������-��
@transformer_block/layer_normalization/batchnorm_4/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5transformer_block/layer_normalization/batchnorm_4/subSubHtransformer_block/layer_normalization/batchnorm_4/ReadVariableOp:value:0;transformer_block/layer_normalization/batchnorm_4/mul_2:z:0*
T0*,
_output_shapes
:���������-��
7transformer_block/layer_normalization/batchnorm_4/add_1AddV2;transformer_block/layer_normalization/batchnorm_4/mul_1:z:09transformer_block/layer_normalization/batchnorm_4/sub:z:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense/Tensordot_4/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0}
3transformer_block/sequential/dense/Tensordot_4/axesConst*
_output_shapes
:*
dtype0*
valueB:�
3transformer_block/sequential/dense/Tensordot_4/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
4transformer_block/sequential/dense/Tensordot_4/ShapeShape;transformer_block/layer_normalization/batchnorm_4/add_1:z:0*
T0*
_output_shapes
:~
<transformer_block/sequential/dense/Tensordot_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot_4/GatherV2GatherV2=transformer_block/sequential/dense/Tensordot_4/Shape:output:0<transformer_block/sequential/dense/Tensordot_4/free:output:0Etransformer_block/sequential/dense/Tensordot_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
>transformer_block/sequential/dense/Tensordot_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense/Tensordot_4/GatherV2_1GatherV2=transformer_block/sequential/dense/Tensordot_4/Shape:output:0<transformer_block/sequential/dense/Tensordot_4/axes:output:0Gtransformer_block/sequential/dense/Tensordot_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4transformer_block/sequential/dense/Tensordot_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
3transformer_block/sequential/dense/Tensordot_4/ProdProd@transformer_block/sequential/dense/Tensordot_4/GatherV2:output:0=transformer_block/sequential/dense/Tensordot_4/Const:output:0*
T0*
_output_shapes
: �
6transformer_block/sequential/dense/Tensordot_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense/Tensordot_4/Prod_1ProdBtransformer_block/sequential/dense/Tensordot_4/GatherV2_1:output:0?transformer_block/sequential/dense/Tensordot_4/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_block/sequential/dense/Tensordot_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5transformer_block/sequential/dense/Tensordot_4/concatConcatV2<transformer_block/sequential/dense/Tensordot_4/free:output:0<transformer_block/sequential/dense/Tensordot_4/axes:output:0Ctransformer_block/sequential/dense/Tensordot_4/concat/axis:output:0*
N*
T0*
_output_shapes
:�
4transformer_block/sequential/dense/Tensordot_4/stackPack<transformer_block/sequential/dense/Tensordot_4/Prod:output:0>transformer_block/sequential/dense/Tensordot_4/Prod_1:output:0*
N*
T0*
_output_shapes
:�
8transformer_block/sequential/dense/Tensordot_4/transpose	Transpose;transformer_block/layer_normalization/batchnorm_4/add_1:z:0>transformer_block/sequential/dense/Tensordot_4/concat:output:0*
T0*,
_output_shapes
:���������-��
6transformer_block/sequential/dense/Tensordot_4/ReshapeReshape<transformer_block/sequential/dense/Tensordot_4/transpose:y:0=transformer_block/sequential/dense/Tensordot_4/stack:output:0*
T0*0
_output_shapes
:�������������������
5transformer_block/sequential/dense/Tensordot_4/MatMulMatMul?transformer_block/sequential/dense/Tensordot_4/Reshape:output:0Etransformer_block/sequential/dense/Tensordot_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
6transformer_block/sequential/dense/Tensordot_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�~
<transformer_block/sequential/dense/Tensordot_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense/Tensordot_4/concat_1ConcatV2@transformer_block/sequential/dense/Tensordot_4/GatherV2:output:0?transformer_block/sequential/dense/Tensordot_4/Const_2:output:0Etransformer_block/sequential/dense/Tensordot_4/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
.transformer_block/sequential/dense/Tensordot_4Reshape?transformer_block/sequential/dense/Tensordot_4/MatMul:product:0@transformer_block/sequential/dense/Tensordot_4/concat_1:output:0*
T0*,
_output_shapes
:���������-��
;transformer_block/sequential/dense/BiasAdd_4/ReadVariableOpReadVariableOpBtransformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,transformer_block/sequential/dense/BiasAdd_4BiasAdd7transformer_block/sequential/dense/Tensordot_4:output:0Ctransformer_block/sequential/dense/BiasAdd_4/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
)transformer_block/sequential/dense/Selu_4Selu5transformer_block/sequential/dense/BiasAdd_4:output:0*
T0*,
_output_shapes
:���������-��
?transformer_block/sequential/dense_1/Tensordot_4/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0
5transformer_block/sequential/dense_1/Tensordot_4/axesConst*
_output_shapes
:*
dtype0*
valueB:�
5transformer_block/sequential/dense_1/Tensordot_4/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
6transformer_block/sequential/dense_1/Tensordot_4/ShapeShape7transformer_block/sequential/dense/Selu_4:activations:0*
T0*
_output_shapes
:�
>transformer_block/sequential/dense_1/Tensordot_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot_4/GatherV2GatherV2?transformer_block/sequential/dense_1/Tensordot_4/Shape:output:0>transformer_block/sequential/dense_1/Tensordot_4/free:output:0Gtransformer_block/sequential/dense_1/Tensordot_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
@transformer_block/sequential/dense_1/Tensordot_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;transformer_block/sequential/dense_1/Tensordot_4/GatherV2_1GatherV2?transformer_block/sequential/dense_1/Tensordot_4/Shape:output:0>transformer_block/sequential/dense_1/Tensordot_4/axes:output:0Itransformer_block/sequential/dense_1/Tensordot_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
6transformer_block/sequential/dense_1/Tensordot_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
5transformer_block/sequential/dense_1/Tensordot_4/ProdProdBtransformer_block/sequential/dense_1/Tensordot_4/GatherV2:output:0?transformer_block/sequential/dense_1/Tensordot_4/Const:output:0*
T0*
_output_shapes
: �
8transformer_block/sequential/dense_1/Tensordot_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
7transformer_block/sequential/dense_1/Tensordot_4/Prod_1ProdDtransformer_block/sequential/dense_1/Tensordot_4/GatherV2_1:output:0Atransformer_block/sequential/dense_1/Tensordot_4/Const_1:output:0*
T0*
_output_shapes
: ~
<transformer_block/sequential/dense_1/Tensordot_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7transformer_block/sequential/dense_1/Tensordot_4/concatConcatV2>transformer_block/sequential/dense_1/Tensordot_4/free:output:0>transformer_block/sequential/dense_1/Tensordot_4/axes:output:0Etransformer_block/sequential/dense_1/Tensordot_4/concat/axis:output:0*
N*
T0*
_output_shapes
:�
6transformer_block/sequential/dense_1/Tensordot_4/stackPack>transformer_block/sequential/dense_1/Tensordot_4/Prod:output:0@transformer_block/sequential/dense_1/Tensordot_4/Prod_1:output:0*
N*
T0*
_output_shapes
:�
:transformer_block/sequential/dense_1/Tensordot_4/transpose	Transpose7transformer_block/sequential/dense/Selu_4:activations:0@transformer_block/sequential/dense_1/Tensordot_4/concat:output:0*
T0*,
_output_shapes
:���������-��
8transformer_block/sequential/dense_1/Tensordot_4/ReshapeReshape>transformer_block/sequential/dense_1/Tensordot_4/transpose:y:0?transformer_block/sequential/dense_1/Tensordot_4/stack:output:0*
T0*0
_output_shapes
:�������������������
7transformer_block/sequential/dense_1/Tensordot_4/MatMulMatMulAtransformer_block/sequential/dense_1/Tensordot_4/Reshape:output:0Gtransformer_block/sequential/dense_1/Tensordot_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
8transformer_block/sequential/dense_1/Tensordot_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
>transformer_block/sequential/dense_1/Tensordot_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9transformer_block/sequential/dense_1/Tensordot_4/concat_1ConcatV2Btransformer_block/sequential/dense_1/Tensordot_4/GatherV2:output:0Atransformer_block/sequential/dense_1/Tensordot_4/Const_2:output:0Gtransformer_block/sequential/dense_1/Tensordot_4/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
0transformer_block/sequential/dense_1/Tensordot_4ReshapeAtransformer_block/sequential/dense_1/Tensordot_4/MatMul:product:0Btransformer_block/sequential/dense_1/Tensordot_4/concat_1:output:0*
T0*,
_output_shapes
:���������-��
=transformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.transformer_block/sequential/dense_1/BiasAdd_4BiasAdd9transformer_block/sequential/dense_1/Tensordot_4:output:0Etransformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�p
+transformer_block/dropout_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
)transformer_block/dropout_1/dropout_4/MulMul7transformer_block/sequential/dense_1/BiasAdd_4:output:04transformer_block/dropout_1/dropout_4/Const:output:0*
T0*,
_output_shapes
:���������-��
+transformer_block/dropout_1/dropout_4/ShapeShape7transformer_block/sequential/dense_1/BiasAdd_4:output:0*
T0*
_output_shapes
:�
Btransformer_block/dropout_1/dropout_4/random_uniform/RandomUniformRandomUniform4transformer_block/dropout_1/dropout_4/Shape:output:0*
T0*,
_output_shapes
:���������-�*
dtype0y
4transformer_block/dropout_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *U;>�
2transformer_block/dropout_1/dropout_4/GreaterEqualGreaterEqualKtransformer_block/dropout_1/dropout_4/random_uniform/RandomUniform:output:0=transformer_block/dropout_1/dropout_4/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������-��
*transformer_block/dropout_1/dropout_4/CastCast6transformer_block/dropout_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������-��
+transformer_block/dropout_1/dropout_4/Mul_1Mul-transformer_block/dropout_1/dropout_4/Mul:z:0.transformer_block/dropout_1/dropout_4/Cast:y:0*
T0*,
_output_shapes
:���������-��
transformer_block/add_9AddV2;transformer_block/layer_normalization/batchnorm_4/add_1:z:0/transformer_block/dropout_1/dropout_4/Mul_1:z:0*
T0*,
_output_shapes
:���������-��
Htransformer_block/layer_normalization_1/moments_4/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
6transformer_block/layer_normalization_1/moments_4/meanMeantransformer_block/add_9:z:0Qtransformer_block/layer_normalization_1/moments_4/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
>transformer_block/layer_normalization_1/moments_4/StopGradientStopGradient?transformer_block/layer_normalization_1/moments_4/mean:output:0*
T0*+
_output_shapes
:���������-�
Ctransformer_block/layer_normalization_1/moments_4/SquaredDifferenceSquaredDifferencetransformer_block/add_9:z:0Gtransformer_block/layer_normalization_1/moments_4/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Ltransformer_block/layer_normalization_1/moments_4/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:transformer_block/layer_normalization_1/moments_4/varianceMeanGtransformer_block/layer_normalization_1/moments_4/SquaredDifference:z:0Utransformer_block/layer_normalization_1/moments_4/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(~
9transformer_block/layer_normalization_1/batchnorm_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
7transformer_block/layer_normalization_1/batchnorm_4/addAddV2Ctransformer_block/layer_normalization_1/moments_4/variance:output:0Btransformer_block/layer_normalization_1/batchnorm_4/add/y:output:0*
T0*+
_output_shapes
:���������-�
9transformer_block/layer_normalization_1/batchnorm_4/RsqrtRsqrt;transformer_block/layer_normalization_1/batchnorm_4/add:z:0*
T0*+
_output_shapes
:���������-�
Ftransformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_block/layer_normalization_1/batchnorm_4/mulMul=transformer_block/layer_normalization_1/batchnorm_4/Rsqrt:y:0Ntransformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_4/mul_1Multransformer_block/add_9:z:0;transformer_block/layer_normalization_1/batchnorm_4/mul:z:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_4/mul_2Mul?transformer_block/layer_normalization_1/moments_4/mean:output:0;transformer_block/layer_normalization_1/batchnorm_4/mul:z:0*
T0*,
_output_shapes
:���������-��
Btransformer_block/layer_normalization_1/batchnorm_4/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_block/layer_normalization_1/batchnorm_4/subSubJtransformer_block/layer_normalization_1/batchnorm_4/ReadVariableOp:value:0=transformer_block/layer_normalization_1/batchnorm_4/mul_2:z:0*
T0*,
_output_shapes
:���������-��
9transformer_block/layer_normalization_1/batchnorm_4/add_1AddV2=transformer_block/layer_normalization_1/batchnorm_4/mul_1:z:0;transformer_block/layer_normalization_1/batchnorm_4/sub:z:0*
T0*,
_output_shapes
:���������-�z
reshape/ShapeShape=transformer_block/layer_normalization_1/batchnorm_4/add_1:z:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Z
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :�-�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
reshape/ReshapeReshape=transformer_block/layer_normalization_1/batchnorm_4/add_1:z:0reshape/Reshape/shape:output:0*
T0*,
_output_shapes
:����������-\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dropout_2/dropout/MulMulreshape/Reshape:output:0 dropout_2/dropout/Const:output:0*
T0*,
_output_shapes
:����������-_
dropout_2/dropout/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:�
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*,
_output_shapes
:����������-*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * gL=�
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������-�
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������-�
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*,
_output_shapes
:����������-�
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
�-�*
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_2/Tensordot/ShapeShapedropout_2/dropout/Mul_1:z:0*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_2/Tensordot/transpose	Transposedropout_2/dropout/Mul_1:z:0!dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������-�
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������e
dense_2/SeluSeludense_2/BiasAdd:output:0*
T0*,
_output_shapes
:����������\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dropout_3/dropout/MulMuldense_2/Selu:activations:0 dropout_3/dropout/Const:output:0*
T0*,
_output_shapes
:����������a
dropout_3/dropout/ShapeShapedense_2/Selu:activations:0*
T0*
_output_shapes
:�
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * gL=�
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:�����������
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:�����������
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*,
_output_shapes
:�����������
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0`
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_3/Tensordot/ShapeShapedropout_3/dropout/Mul_1:z:0*
T0*
_output_shapes
:a
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_3/Tensordot/transpose	Transposedropout_3/dropout/Mul_1:z:0!dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������e
dense_3/SeluSeludense_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dropout_4/dropout/MulMuldense_3/Selu:activations:0 dropout_4/dropout/Const:output:0*
T0*,
_output_shapes
:����������a
dropout_4/dropout/ShapeShapedense_3/Selu:activations:0*
T0*
_output_shapes
:�
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * gL=�
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:�����������
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:�����������
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*,
_output_shapes
:�����������
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_4/Tensordot/ShapeShapedropout_4/dropout/Mul_1:z:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_4/Tensordot/transpose	Transposedropout_4/dropout/Mul_1:z:0!dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������@�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@d
dense_4/SeluSeludense_4/BiasAdd:output:0*
T0*+
_output_shapes
:���������@\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dropout_5/dropout/MulMuldense_4/Selu:activations:0 dropout_5/dropout/Const:output:0*
T0*+
_output_shapes
:���������@a
dropout_5/dropout/ShapeShapedense_4/Selu:activations:0*
T0*
_output_shapes
:�
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*+
_output_shapes
:���������@*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * gL=�
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������@�
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������@�
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*+
_output_shapes
:���������@�
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0`
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_5/Tensordot/ShapeShapedropout_5/dropout/Mul_1:z:0*
T0*
_output_shapes
:a
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_5/Tensordot/transpose	Transposedropout_5/dropout/Mul_1:z:0!dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������@�
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������c
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:����������
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������d
dense_5/SeluSeludense_5/BiasAdd:output:0*
T0*+
_output_shapes
:���������\
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dropout_6/dropout/MulMuldense_5/Selu:activations:0 dropout_6/dropout/Const:output:0*
T0*+
_output_shapes
:���������a
dropout_6/dropout/ShapeShapedense_5/Selu:activations:0*
T0*
_output_shapes
:�
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*+
_output_shapes
:���������*
dtype0e
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * gL=�
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:����������
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:����������
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*+
_output_shapes
:����������
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_6/Tensordot/ShapeShapedropout_6/dropout/Mul_1:z:0*
T0*
_output_shapes
:a
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_6/Tensordot/transpose	Transposedropout_6/dropout/Mul_1:z:0!dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:����������
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������c
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:����������
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������d
dense_6/SeluSeludense_6/BiasAdd:output:0*
T0*+
_output_shapes
:���������m
IdentityIdentitydense_6/Selu:activations:0^NoOp*
T0*+
_output_shapes
:����������0
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/Tensordot/ReadVariableOp8^token_and_position_embedding/embedding/embedding_lookup:^token_and_position_embedding/embedding_1/embedding_lookup?^transformer_block/layer_normalization/batchnorm/ReadVariableOpC^transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpA^transformer_block/layer_normalization/batchnorm_1/ReadVariableOpE^transformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOpA^transformer_block/layer_normalization/batchnorm_2/ReadVariableOpE^transformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOpA^transformer_block/layer_normalization/batchnorm_3/ReadVariableOpE^transformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOpA^transformer_block/layer_normalization/batchnorm_4/ReadVariableOpE^transformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOpA^transformer_block/layer_normalization_1/batchnorm/ReadVariableOpE^transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpC^transformer_block/layer_normalization_1/batchnorm_1/ReadVariableOpG^transformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOpC^transformer_block/layer_normalization_1/batchnorm_2/ReadVariableOpG^transformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOpC^transformer_block/layer_normalization_1/batchnorm_3/ReadVariableOpG^transformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOpC^transformer_block/layer_normalization_1/batchnorm_4/ReadVariableOpG^transformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOpK^transformer_block/multi_head_attention/attention_output/add/ReadVariableOpM^transformer_block/multi_head_attention/attention_output/add_1/ReadVariableOpM^transformer_block/multi_head_attention/attention_output/add_2/ReadVariableOpM^transformer_block/multi_head_attention/attention_output/add_3/ReadVariableOpM^transformer_block/multi_head_attention/attention_output/add_4/ReadVariableOpU^transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpW^transformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOpW^transformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOpW^transformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOpW^transformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOp>^transformer_block/multi_head_attention/key/add/ReadVariableOp@^transformer_block/multi_head_attention/key/add_1/ReadVariableOp@^transformer_block/multi_head_attention/key/add_2/ReadVariableOp@^transformer_block/multi_head_attention/key/add_3/ReadVariableOp@^transformer_block/multi_head_attention/key/add_4/ReadVariableOpH^transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpJ^transformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOpJ^transformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOpJ^transformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOpJ^transformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOp@^transformer_block/multi_head_attention/query/add/ReadVariableOpB^transformer_block/multi_head_attention/query/add_1/ReadVariableOpB^transformer_block/multi_head_attention/query/add_2/ReadVariableOpB^transformer_block/multi_head_attention/query/add_3/ReadVariableOpB^transformer_block/multi_head_attention/query/add_4/ReadVariableOpJ^transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpL^transformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOpL^transformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOpL^transformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOpL^transformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOp@^transformer_block/multi_head_attention/value/add/ReadVariableOpB^transformer_block/multi_head_attention/value/add_1/ReadVariableOpB^transformer_block/multi_head_attention/value/add_2/ReadVariableOpB^transformer_block/multi_head_attention/value/add_3/ReadVariableOpB^transformer_block/multi_head_attention/value/add_4/ReadVariableOpJ^transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpL^transformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOpL^transformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOpL^transformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOpL^transformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOp:^transformer_block/sequential/dense/BiasAdd/ReadVariableOp<^transformer_block/sequential/dense/BiasAdd_1/ReadVariableOp<^transformer_block/sequential/dense/BiasAdd_2/ReadVariableOp<^transformer_block/sequential/dense/BiasAdd_3/ReadVariableOp<^transformer_block/sequential/dense/BiasAdd_4/ReadVariableOp<^transformer_block/sequential/dense/Tensordot/ReadVariableOp>^transformer_block/sequential/dense/Tensordot_1/ReadVariableOp>^transformer_block/sequential/dense/Tensordot_2/ReadVariableOp>^transformer_block/sequential/dense/Tensordot_3/ReadVariableOp>^transformer_block/sequential/dense/Tensordot_4/ReadVariableOp<^transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp>^transformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOp>^transformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOp>^transformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOp>^transformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOp>^transformer_block/sequential/dense_1/Tensordot/ReadVariableOp@^transformer_block/sequential/dense_1/Tensordot_1/ReadVariableOp@^transformer_block/sequential/dense_1/Tensordot_2/ReadVariableOp@^transformer_block/sequential/dense_1/Tensordot_3/ReadVariableOp@^transformer_block/sequential/dense_1/Tensordot_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������-: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2D
 dense_6/Tensordot/ReadVariableOp dense_6/Tensordot/ReadVariableOp2r
7token_and_position_embedding/embedding/embedding_lookup7token_and_position_embedding/embedding/embedding_lookup2v
9token_and_position_embedding/embedding_1/embedding_lookup9token_and_position_embedding/embedding_1/embedding_lookup2�
>transformer_block/layer_normalization/batchnorm/ReadVariableOp>transformer_block/layer_normalization/batchnorm/ReadVariableOp2�
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpBtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2�
@transformer_block/layer_normalization/batchnorm_1/ReadVariableOp@transformer_block/layer_normalization/batchnorm_1/ReadVariableOp2�
Dtransformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOpDtransformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOp2�
@transformer_block/layer_normalization/batchnorm_2/ReadVariableOp@transformer_block/layer_normalization/batchnorm_2/ReadVariableOp2�
Dtransformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOpDtransformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOp2�
@transformer_block/layer_normalization/batchnorm_3/ReadVariableOp@transformer_block/layer_normalization/batchnorm_3/ReadVariableOp2�
Dtransformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOpDtransformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOp2�
@transformer_block/layer_normalization/batchnorm_4/ReadVariableOp@transformer_block/layer_normalization/batchnorm_4/ReadVariableOp2�
Dtransformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOpDtransformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOp2�
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2�
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpDtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2�
Btransformer_block/layer_normalization_1/batchnorm_1/ReadVariableOpBtransformer_block/layer_normalization_1/batchnorm_1/ReadVariableOp2�
Ftransformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOpFtransformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOp2�
Btransformer_block/layer_normalization_1/batchnorm_2/ReadVariableOpBtransformer_block/layer_normalization_1/batchnorm_2/ReadVariableOp2�
Ftransformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOpFtransformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOp2�
Btransformer_block/layer_normalization_1/batchnorm_3/ReadVariableOpBtransformer_block/layer_normalization_1/batchnorm_3/ReadVariableOp2�
Ftransformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOpFtransformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOp2�
Btransformer_block/layer_normalization_1/batchnorm_4/ReadVariableOpBtransformer_block/layer_normalization_1/batchnorm_4/ReadVariableOp2�
Ftransformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOpFtransformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOp2�
Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOpJtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp2�
Ltransformer_block/multi_head_attention/attention_output/add_1/ReadVariableOpLtransformer_block/multi_head_attention/attention_output/add_1/ReadVariableOp2�
Ltransformer_block/multi_head_attention/attention_output/add_2/ReadVariableOpLtransformer_block/multi_head_attention/attention_output/add_2/ReadVariableOp2�
Ltransformer_block/multi_head_attention/attention_output/add_3/ReadVariableOpLtransformer_block/multi_head_attention/attention_output/add_3/ReadVariableOp2�
Ltransformer_block/multi_head_attention/attention_output/add_4/ReadVariableOpLtransformer_block/multi_head_attention/attention_output/add_4/ReadVariableOp2�
Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpTtransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2�
Vtransformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOpVtransformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOp2�
Vtransformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOpVtransformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOp2�
Vtransformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOpVtransformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOp2�
Vtransformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOpVtransformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOp2~
=transformer_block/multi_head_attention/key/add/ReadVariableOp=transformer_block/multi_head_attention/key/add/ReadVariableOp2�
?transformer_block/multi_head_attention/key/add_1/ReadVariableOp?transformer_block/multi_head_attention/key/add_1/ReadVariableOp2�
?transformer_block/multi_head_attention/key/add_2/ReadVariableOp?transformer_block/multi_head_attention/key/add_2/ReadVariableOp2�
?transformer_block/multi_head_attention/key/add_3/ReadVariableOp?transformer_block/multi_head_attention/key/add_3/ReadVariableOp2�
?transformer_block/multi_head_attention/key/add_4/ReadVariableOp?transformer_block/multi_head_attention/key/add_4/ReadVariableOp2�
Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpGtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp2�
Itransformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOpItransformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOp2�
Itransformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOpItransformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOp2�
Itransformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOpItransformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOp2�
Itransformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOpItransformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOp2�
?transformer_block/multi_head_attention/query/add/ReadVariableOp?transformer_block/multi_head_attention/query/add/ReadVariableOp2�
Atransformer_block/multi_head_attention/query/add_1/ReadVariableOpAtransformer_block/multi_head_attention/query/add_1/ReadVariableOp2�
Atransformer_block/multi_head_attention/query/add_2/ReadVariableOpAtransformer_block/multi_head_attention/query/add_2/ReadVariableOp2�
Atransformer_block/multi_head_attention/query/add_3/ReadVariableOpAtransformer_block/multi_head_attention/query/add_3/ReadVariableOp2�
Atransformer_block/multi_head_attention/query/add_4/ReadVariableOpAtransformer_block/multi_head_attention/query/add_4/ReadVariableOp2�
Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpItransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp2�
Ktransformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOpKtransformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOp2�
Ktransformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOpKtransformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOp2�
Ktransformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOpKtransformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOp2�
Ktransformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOpKtransformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOp2�
?transformer_block/multi_head_attention/value/add/ReadVariableOp?transformer_block/multi_head_attention/value/add/ReadVariableOp2�
Atransformer_block/multi_head_attention/value/add_1/ReadVariableOpAtransformer_block/multi_head_attention/value/add_1/ReadVariableOp2�
Atransformer_block/multi_head_attention/value/add_2/ReadVariableOpAtransformer_block/multi_head_attention/value/add_2/ReadVariableOp2�
Atransformer_block/multi_head_attention/value/add_3/ReadVariableOpAtransformer_block/multi_head_attention/value/add_3/ReadVariableOp2�
Atransformer_block/multi_head_attention/value/add_4/ReadVariableOpAtransformer_block/multi_head_attention/value/add_4/ReadVariableOp2�
Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpItransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp2�
Ktransformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOpKtransformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOp2�
Ktransformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOpKtransformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOp2�
Ktransformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOpKtransformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOp2�
Ktransformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOpKtransformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOp2v
9transformer_block/sequential/dense/BiasAdd/ReadVariableOp9transformer_block/sequential/dense/BiasAdd/ReadVariableOp2z
;transformer_block/sequential/dense/BiasAdd_1/ReadVariableOp;transformer_block/sequential/dense/BiasAdd_1/ReadVariableOp2z
;transformer_block/sequential/dense/BiasAdd_2/ReadVariableOp;transformer_block/sequential/dense/BiasAdd_2/ReadVariableOp2z
;transformer_block/sequential/dense/BiasAdd_3/ReadVariableOp;transformer_block/sequential/dense/BiasAdd_3/ReadVariableOp2z
;transformer_block/sequential/dense/BiasAdd_4/ReadVariableOp;transformer_block/sequential/dense/BiasAdd_4/ReadVariableOp2z
;transformer_block/sequential/dense/Tensordot/ReadVariableOp;transformer_block/sequential/dense/Tensordot/ReadVariableOp2~
=transformer_block/sequential/dense/Tensordot_1/ReadVariableOp=transformer_block/sequential/dense/Tensordot_1/ReadVariableOp2~
=transformer_block/sequential/dense/Tensordot_2/ReadVariableOp=transformer_block/sequential/dense/Tensordot_2/ReadVariableOp2~
=transformer_block/sequential/dense/Tensordot_3/ReadVariableOp=transformer_block/sequential/dense/Tensordot_3/ReadVariableOp2~
=transformer_block/sequential/dense/Tensordot_4/ReadVariableOp=transformer_block/sequential/dense/Tensordot_4/ReadVariableOp2z
;transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp;transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp2~
=transformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOp=transformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOp2~
=transformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOp=transformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOp2~
=transformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOp=transformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOp2~
=transformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOp=transformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOp2~
=transformer_block/sequential/dense_1/Tensordot/ReadVariableOp=transformer_block/sequential/dense_1/Tensordot/ReadVariableOp2�
?transformer_block/sequential/dense_1/Tensordot_1/ReadVariableOp?transformer_block/sequential/dense_1/Tensordot_1/ReadVariableOp2�
?transformer_block/sequential/dense_1/Tensordot_2/ReadVariableOp?transformer_block/sequential/dense_1/Tensordot_2/ReadVariableOp2�
?transformer_block/sequential/dense_1/Tensordot_3/ReadVariableOp?transformer_block/sequential/dense_1/Tensordot_3/ReadVariableOp2�
?transformer_block/sequential/dense_1/Tensordot_4/ReadVariableOp?transformer_block/sequential/dense_1/Tensordot_4/ReadVariableOp:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�
�
&__inference_dense_2_layer_call_fn_5808

inputs
unknown:
�-�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_2403t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������-: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������-
 
_user_specified_nameinputs
��

�G
__inference__wrapped_model_1891
input_1W
Dmodel_token_and_position_embedding_embedding_1_embedding_lookup_1178:	-�V
Bmodel_token_and_position_embedding_embedding_embedding_lookup_1184:
��p
Xmodel_transformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource:��a
Nmodel_transformer_block_multi_head_attention_query_add_readvariableop_resource:	�n
Vmodel_transformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource:��_
Lmodel_transformer_block_multi_head_attention_key_add_readvariableop_resource:	�p
Xmodel_transformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource:��a
Nmodel_transformer_block_multi_head_attention_value_add_readvariableop_resource:	�{
cmodel_transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:��h
Ymodel_transformer_block_multi_head_attention_attention_output_add_readvariableop_resource:	�`
Qmodel_transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource:	�\
Mmodel_transformer_block_layer_normalization_batchnorm_readvariableop_resource:	�^
Jmodel_transformer_block_sequential_dense_tensordot_readvariableop_resource:
��W
Hmodel_transformer_block_sequential_dense_biasadd_readvariableop_resource:	�`
Lmodel_transformer_block_sequential_dense_1_tensordot_readvariableop_resource:
��Y
Jmodel_transformer_block_sequential_dense_1_biasadd_readvariableop_resource:	�b
Smodel_transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource:	�^
Omodel_transformer_block_layer_normalization_1_batchnorm_readvariableop_resource:	�C
/model_dense_2_tensordot_readvariableop_resource:
�-�<
-model_dense_2_biasadd_readvariableop_resource:	�C
/model_dense_3_tensordot_readvariableop_resource:
��<
-model_dense_3_biasadd_readvariableop_resource:	�B
/model_dense_4_tensordot_readvariableop_resource:	�@;
-model_dense_4_biasadd_readvariableop_resource:@A
/model_dense_5_tensordot_readvariableop_resource:@;
-model_dense_5_biasadd_readvariableop_resource:A
/model_dense_6_tensordot_readvariableop_resource:;
-model_dense_6_biasadd_readvariableop_resource:
identity��$model/dense_2/BiasAdd/ReadVariableOp�&model/dense_2/Tensordot/ReadVariableOp�$model/dense_3/BiasAdd/ReadVariableOp�&model/dense_3/Tensordot/ReadVariableOp�$model/dense_4/BiasAdd/ReadVariableOp�&model/dense_4/Tensordot/ReadVariableOp�$model/dense_5/BiasAdd/ReadVariableOp�&model/dense_5/Tensordot/ReadVariableOp�$model/dense_6/BiasAdd/ReadVariableOp�&model/dense_6/Tensordot/ReadVariableOp�=model/token_and_position_embedding/embedding/embedding_lookup�?model/token_and_position_embedding/embedding_1/embedding_lookup�Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp�Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp�Fmodel/transformer_block/layer_normalization/batchnorm_1/ReadVariableOp�Jmodel/transformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOp�Fmodel/transformer_block/layer_normalization/batchnorm_2/ReadVariableOp�Jmodel/transformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOp�Fmodel/transformer_block/layer_normalization/batchnorm_3/ReadVariableOp�Jmodel/transformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOp�Fmodel/transformer_block/layer_normalization/batchnorm_4/ReadVariableOp�Jmodel/transformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOp�Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp�Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp�Hmodel/transformer_block/layer_normalization_1/batchnorm_1/ReadVariableOp�Lmodel/transformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOp�Hmodel/transformer_block/layer_normalization_1/batchnorm_2/ReadVariableOp�Lmodel/transformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOp�Hmodel/transformer_block/layer_normalization_1/batchnorm_3/ReadVariableOp�Lmodel/transformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOp�Hmodel/transformer_block/layer_normalization_1/batchnorm_4/ReadVariableOp�Lmodel/transformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOp�Pmodel/transformer_block/multi_head_attention/attention_output/add/ReadVariableOp�Rmodel/transformer_block/multi_head_attention/attention_output/add_1/ReadVariableOp�Rmodel/transformer_block/multi_head_attention/attention_output/add_2/ReadVariableOp�Rmodel/transformer_block/multi_head_attention/attention_output/add_3/ReadVariableOp�Rmodel/transformer_block/multi_head_attention/attention_output/add_4/ReadVariableOp�Zmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp�\model/transformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOp�\model/transformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOp�\model/transformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOp�\model/transformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOp�Cmodel/transformer_block/multi_head_attention/key/add/ReadVariableOp�Emodel/transformer_block/multi_head_attention/key/add_1/ReadVariableOp�Emodel/transformer_block/multi_head_attention/key/add_2/ReadVariableOp�Emodel/transformer_block/multi_head_attention/key/add_3/ReadVariableOp�Emodel/transformer_block/multi_head_attention/key/add_4/ReadVariableOp�Mmodel/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp�Omodel/transformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOp�Omodel/transformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOp�Omodel/transformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOp�Omodel/transformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOp�Emodel/transformer_block/multi_head_attention/query/add/ReadVariableOp�Gmodel/transformer_block/multi_head_attention/query/add_1/ReadVariableOp�Gmodel/transformer_block/multi_head_attention/query/add_2/ReadVariableOp�Gmodel/transformer_block/multi_head_attention/query/add_3/ReadVariableOp�Gmodel/transformer_block/multi_head_attention/query/add_4/ReadVariableOp�Omodel/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp�Qmodel/transformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOp�Qmodel/transformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOp�Qmodel/transformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOp�Qmodel/transformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOp�Emodel/transformer_block/multi_head_attention/value/add/ReadVariableOp�Gmodel/transformer_block/multi_head_attention/value/add_1/ReadVariableOp�Gmodel/transformer_block/multi_head_attention/value/add_2/ReadVariableOp�Gmodel/transformer_block/multi_head_attention/value/add_3/ReadVariableOp�Gmodel/transformer_block/multi_head_attention/value/add_4/ReadVariableOp�Omodel/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp�Qmodel/transformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOp�Qmodel/transformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOp�Qmodel/transformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOp�Qmodel/transformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOp�?model/transformer_block/sequential/dense/BiasAdd/ReadVariableOp�Amodel/transformer_block/sequential/dense/BiasAdd_1/ReadVariableOp�Amodel/transformer_block/sequential/dense/BiasAdd_2/ReadVariableOp�Amodel/transformer_block/sequential/dense/BiasAdd_3/ReadVariableOp�Amodel/transformer_block/sequential/dense/BiasAdd_4/ReadVariableOp�Amodel/transformer_block/sequential/dense/Tensordot/ReadVariableOp�Cmodel/transformer_block/sequential/dense/Tensordot_1/ReadVariableOp�Cmodel/transformer_block/sequential/dense/Tensordot_2/ReadVariableOp�Cmodel/transformer_block/sequential/dense/Tensordot_3/ReadVariableOp�Cmodel/transformer_block/sequential/dense/Tensordot_4/ReadVariableOp�Amodel/transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp�Cmodel/transformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOp�Cmodel/transformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOp�Cmodel/transformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOp�Cmodel/transformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOp�Cmodel/transformer_block/sequential/dense_1/Tensordot/ReadVariableOp�Emodel/transformer_block/sequential/dense_1/Tensordot_1/ReadVariableOp�Emodel/transformer_block/sequential/dense_1/Tensordot_2/ReadVariableOp�Emodel/transformer_block/sequential/dense_1/Tensordot_3/ReadVariableOp�Emodel/transformer_block/sequential/dense_1/Tensordot_4/ReadVariableOp_
(model/token_and_position_embedding/ShapeShapeinput_1*
T0*
_output_shapes
:�
6model/token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
8model/token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
8model/token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0model/token_and_position_embedding/strided_sliceStridedSlice1model/token_and_position_embedding/Shape:output:0?model/token_and_position_embedding/strided_slice/stack:output:0Amodel/token_and_position_embedding/strided_slice/stack_1:output:0Amodel/token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.model/token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : p
.model/token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
(model/token_and_position_embedding/rangeRange7model/token_and_position_embedding/range/start:output:09model/token_and_position_embedding/strided_slice:output:07model/token_and_position_embedding/range/delta:output:0*
_output_shapes
:-�
?model/token_and_position_embedding/embedding_1/embedding_lookupResourceGatherDmodel_token_and_position_embedding_embedding_1_embedding_lookup_11781model/token_and_position_embedding/range:output:0*
Tindices0*W
_classM
KIloc:@model/token_and_position_embedding/embedding_1/embedding_lookup/1178*
_output_shapes
:	-�*
dtype0�
Hmodel/token_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityHmodel/token_and_position_embedding/embedding_1/embedding_lookup:output:0*
T0*W
_classM
KIloc:@model/token_and_position_embedding/embedding_1/embedding_lookup/1178*
_output_shapes
:	-��
Jmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityQmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	-��
1model/token_and_position_embedding/embedding/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:���������-�
=model/token_and_position_embedding/embedding/embedding_lookupResourceGatherBmodel_token_and_position_embedding_embedding_embedding_lookup_11845model/token_and_position_embedding/embedding/Cast:y:0*
Tindices0*U
_classK
IGloc:@model/token_and_position_embedding/embedding/embedding_lookup/1184*,
_output_shapes
:���������-�*
dtype0�
Fmodel/token_and_position_embedding/embedding/embedding_lookup/IdentityIdentityFmodel/token_and_position_embedding/embedding/embedding_lookup:output:0*
T0*U
_classK
IGloc:@model/token_and_position_embedding/embedding/embedding_lookup/1184*,
_output_shapes
:���������-��
Hmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityOmodel/token_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:���������-��
&model/token_and_position_embedding/addAddV2Qmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Smodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:���������-��
Omodel/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpXmodel_transformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
@model/transformer_block/multi_head_attention/query/einsum/EinsumEinsum*model/token_and_position_embedding/add:z:0Wmodel/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Emodel/transformer_block/multi_head_attention/query/add/ReadVariableOpReadVariableOpNmodel_transformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
6model/transformer_block/multi_head_attention/query/addAddV2Imodel/transformer_block/multi_head_attention/query/einsum/Einsum:output:0Mmodel/transformer_block/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Mmodel/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpVmodel_transformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
>model/transformer_block/multi_head_attention/key/einsum/EinsumEinsum*model/token_and_position_embedding/add:z:0Umodel/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Cmodel/transformer_block/multi_head_attention/key/add/ReadVariableOpReadVariableOpLmodel_transformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
4model/transformer_block/multi_head_attention/key/addAddV2Gmodel/transformer_block/multi_head_attention/key/einsum/Einsum:output:0Kmodel/transformer_block/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Omodel/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpXmodel_transformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
@model/transformer_block/multi_head_attention/value/einsum/EinsumEinsum*model/token_and_position_embedding/add:z:0Wmodel/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Emodel/transformer_block/multi_head_attention/value/add/ReadVariableOpReadVariableOpNmodel_transformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
6model/transformer_block/multi_head_attention/value/addAddV2Imodel/transformer_block/multi_head_attention/value/einsum/Einsum:output:0Mmodel/transformer_block/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�w
2model/transformer_block/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
0model/transformer_block/multi_head_attention/MulMul:model/transformer_block/multi_head_attention/query/add:z:0;model/transformer_block/multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:���������-��
:model/transformer_block/multi_head_attention/einsum/EinsumEinsum8model/transformer_block/multi_head_attention/key/add:z:04model/transformer_block/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
<model/transformer_block/multi_head_attention/softmax/SoftmaxSoftmaxCmodel/transformer_block/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������--�
=model/transformer_block/multi_head_attention/dropout/IdentityIdentityFmodel/transformer_block/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������--�
<model/transformer_block/multi_head_attention/einsum_1/EinsumEinsumFmodel/transformer_block/multi_head_attention/dropout/Identity:output:0:model/transformer_block/multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
Zmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpcmodel_transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Kmodel/transformer_block/multi_head_attention/attention_output/einsum/EinsumEinsumEmodel/transformer_block/multi_head_attention/einsum_1/Einsum:output:0bmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
Pmodel/transformer_block/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpYmodel_transformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Amodel/transformer_block/multi_head_attention/attention_output/addAddV2Tmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum:output:0Xmodel/transformer_block/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
(model/transformer_block/dropout/IdentityIdentityEmodel/transformer_block/multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:���������-��
model/transformer_block/addAddV2*model/token_and_position_embedding/add:z:01model/transformer_block/dropout/Identity:output:0*
T0*,
_output_shapes
:���������-��
Jmodel/transformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
8model/transformer_block/layer_normalization/moments/meanMeanmodel/transformer_block/add:z:0Smodel/transformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
@model/transformer_block/layer_normalization/moments/StopGradientStopGradientAmodel/transformer_block/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:���������-�
Emodel/transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferencemodel/transformer_block/add:z:0Imodel/transformer_block/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Nmodel/transformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
<model/transformer_block/layer_normalization/moments/varianceMeanImodel/transformer_block/layer_normalization/moments/SquaredDifference:z:0Wmodel/transformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
;model/transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
9model/transformer_block/layer_normalization/batchnorm/addAddV2Emodel/transformer_block/layer_normalization/moments/variance:output:0Dmodel/transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������-�
;model/transformer_block/layer_normalization/batchnorm/RsqrtRsqrt=model/transformer_block/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:���������-�
Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpQmodel_transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
9model/transformer_block/layer_normalization/batchnorm/mulMul?model/transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Pmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
;model/transformer_block/layer_normalization/batchnorm/mul_1Mulmodel/transformer_block/add:z:0=model/transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
;model/transformer_block/layer_normalization/batchnorm/mul_2MulAmodel/transformer_block/layer_normalization/moments/mean:output:0=model/transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpMmodel_transformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
9model/transformer_block/layer_normalization/batchnorm/subSubLmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp:value:0?model/transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������-��
;model/transformer_block/layer_normalization/batchnorm/add_1AddV2?model/transformer_block/layer_normalization/batchnorm/mul_1:z:0=model/transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������-��
Amodel/transformer_block/sequential/dense/Tensordot/ReadVariableOpReadVariableOpJmodel_transformer_block_sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7model/transformer_block/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7model/transformer_block/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
8model/transformer_block/sequential/dense/Tensordot/ShapeShape?model/transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
@model/transformer_block/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;model/transformer_block/sequential/dense/Tensordot/GatherV2GatherV2Amodel/transformer_block/sequential/dense/Tensordot/Shape:output:0@model/transformer_block/sequential/dense/Tensordot/free:output:0Imodel/transformer_block/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmodel/transformer_block/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_block/sequential/dense/Tensordot/GatherV2_1GatherV2Amodel/transformer_block/sequential/dense/Tensordot/Shape:output:0@model/transformer_block/sequential/dense/Tensordot/axes:output:0Kmodel/transformer_block/sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8model/transformer_block/sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7model/transformer_block/sequential/dense/Tensordot/ProdProdDmodel/transformer_block/sequential/dense/Tensordot/GatherV2:output:0Amodel/transformer_block/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:model/transformer_block/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9model/transformer_block/sequential/dense/Tensordot/Prod_1ProdFmodel/transformer_block/sequential/dense/Tensordot/GatherV2_1:output:0Cmodel/transformer_block/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>model/transformer_block/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9model/transformer_block/sequential/dense/Tensordot/concatConcatV2@model/transformer_block/sequential/dense/Tensordot/free:output:0@model/transformer_block/sequential/dense/Tensordot/axes:output:0Gmodel/transformer_block/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8model/transformer_block/sequential/dense/Tensordot/stackPack@model/transformer_block/sequential/dense/Tensordot/Prod:output:0Bmodel/transformer_block/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<model/transformer_block/sequential/dense/Tensordot/transpose	Transpose?model/transformer_block/layer_normalization/batchnorm/add_1:z:0Bmodel/transformer_block/sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
:model/transformer_block/sequential/dense/Tensordot/ReshapeReshape@model/transformer_block/sequential/dense/Tensordot/transpose:y:0Amodel/transformer_block/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9model/transformer_block/sequential/dense/Tensordot/MatMulMatMulCmodel/transformer_block/sequential/dense/Tensordot/Reshape:output:0Imodel/transformer_block/sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:model/transformer_block/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@model/transformer_block/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;model/transformer_block/sequential/dense/Tensordot/concat_1ConcatV2Dmodel/transformer_block/sequential/dense/Tensordot/GatherV2:output:0Cmodel/transformer_block/sequential/dense/Tensordot/Const_2:output:0Imodel/transformer_block/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2model/transformer_block/sequential/dense/TensordotReshapeCmodel/transformer_block/sequential/dense/Tensordot/MatMul:product:0Dmodel/transformer_block/sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-��
?model/transformer_block/sequential/dense/BiasAdd/ReadVariableOpReadVariableOpHmodel_transformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0model/transformer_block/sequential/dense/BiasAddBiasAdd;model/transformer_block/sequential/dense/Tensordot:output:0Gmodel/transformer_block/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
-model/transformer_block/sequential/dense/SeluSelu9model/transformer_block/sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:���������-��
Cmodel/transformer_block/sequential/dense_1/Tensordot/ReadVariableOpReadVariableOpLmodel_transformer_block_sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
9model/transformer_block/sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
9model/transformer_block/sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
:model/transformer_block/sequential/dense_1/Tensordot/ShapeShape;model/transformer_block/sequential/dense/Selu:activations:0*
T0*
_output_shapes
:�
Bmodel/transformer_block/sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_block/sequential/dense_1/Tensordot/GatherV2GatherV2Cmodel/transformer_block/sequential/dense_1/Tensordot/Shape:output:0Bmodel/transformer_block/sequential/dense_1/Tensordot/free:output:0Kmodel/transformer_block/sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Dmodel/transformer_block/sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?model/transformer_block/sequential/dense_1/Tensordot/GatherV2_1GatherV2Cmodel/transformer_block/sequential/dense_1/Tensordot/Shape:output:0Bmodel/transformer_block/sequential/dense_1/Tensordot/axes:output:0Mmodel/transformer_block/sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
:model/transformer_block/sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
9model/transformer_block/sequential/dense_1/Tensordot/ProdProdFmodel/transformer_block/sequential/dense_1/Tensordot/GatherV2:output:0Cmodel/transformer_block/sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
<model/transformer_block/sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
;model/transformer_block/sequential/dense_1/Tensordot/Prod_1ProdHmodel/transformer_block/sequential/dense_1/Tensordot/GatherV2_1:output:0Emodel/transformer_block/sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
@model/transformer_block/sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;model/transformer_block/sequential/dense_1/Tensordot/concatConcatV2Bmodel/transformer_block/sequential/dense_1/Tensordot/free:output:0Bmodel/transformer_block/sequential/dense_1/Tensordot/axes:output:0Imodel/transformer_block/sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
:model/transformer_block/sequential/dense_1/Tensordot/stackPackBmodel/transformer_block/sequential/dense_1/Tensordot/Prod:output:0Dmodel/transformer_block/sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
>model/transformer_block/sequential/dense_1/Tensordot/transpose	Transpose;model/transformer_block/sequential/dense/Selu:activations:0Dmodel/transformer_block/sequential/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
<model/transformer_block/sequential/dense_1/Tensordot/ReshapeReshapeBmodel/transformer_block/sequential/dense_1/Tensordot/transpose:y:0Cmodel/transformer_block/sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
;model/transformer_block/sequential/dense_1/Tensordot/MatMulMatMulEmodel/transformer_block/sequential/dense_1/Tensordot/Reshape:output:0Kmodel/transformer_block/sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
<model/transformer_block/sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Bmodel/transformer_block/sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_block/sequential/dense_1/Tensordot/concat_1ConcatV2Fmodel/transformer_block/sequential/dense_1/Tensordot/GatherV2:output:0Emodel/transformer_block/sequential/dense_1/Tensordot/Const_2:output:0Kmodel/transformer_block/sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
4model/transformer_block/sequential/dense_1/TensordotReshapeEmodel/transformer_block/sequential/dense_1/Tensordot/MatMul:product:0Fmodel/transformer_block/sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-��
Amodel/transformer_block/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpJmodel_transformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
2model/transformer_block/sequential/dense_1/BiasAddBiasAdd=model/transformer_block/sequential/dense_1/Tensordot:output:0Imodel/transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
*model/transformer_block/dropout_1/IdentityIdentity;model/transformer_block/sequential/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������-��
model/transformer_block/add_1AddV2?model/transformer_block/layer_normalization/batchnorm/add_1:z:03model/transformer_block/dropout_1/Identity:output:0*
T0*,
_output_shapes
:���������-��
Lmodel/transformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:model/transformer_block/layer_normalization_1/moments/meanMean!model/transformer_block/add_1:z:0Umodel/transformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
Bmodel/transformer_block/layer_normalization_1/moments/StopGradientStopGradientCmodel/transformer_block/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:���������-�
Gmodel/transformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifference!model/transformer_block/add_1:z:0Kmodel/transformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Pmodel/transformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
>model/transformer_block/layer_normalization_1/moments/varianceMeanKmodel/transformer_block/layer_normalization_1/moments/SquaredDifference:z:0Ymodel/transformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
=model/transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
;model/transformer_block/layer_normalization_1/batchnorm/addAddV2Gmodel/transformer_block/layer_normalization_1/moments/variance:output:0Fmodel/transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������-�
=model/transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt?model/transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:���������-�
Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpSmodel_transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;model/transformer_block/layer_normalization_1/batchnorm/mulMulAmodel/transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Rmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
=model/transformer_block/layer_normalization_1/batchnorm/mul_1Mul!model/transformer_block/add_1:z:0?model/transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
=model/transformer_block/layer_normalization_1/batchnorm/mul_2MulCmodel/transformer_block/layer_normalization_1/moments/mean:output:0?model/transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpOmodel_transformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;model/transformer_block/layer_normalization_1/batchnorm/subSubNmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0Amodel/transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������-��
=model/transformer_block/layer_normalization_1/batchnorm/add_1AddV2Amodel/transformer_block/layer_normalization_1/batchnorm/mul_1:z:0?model/transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������-��
Qmodel/transformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOpReadVariableOpXmodel_transformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Bmodel/transformer_block/multi_head_attention/query/einsum_1/EinsumEinsumAmodel/transformer_block/layer_normalization_1/batchnorm/add_1:z:0Ymodel/transformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Gmodel/transformer_block/multi_head_attention/query/add_1/ReadVariableOpReadVariableOpNmodel_transformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
8model/transformer_block/multi_head_attention/query/add_1AddV2Kmodel/transformer_block/multi_head_attention/query/einsum_1/Einsum:output:0Omodel/transformer_block/multi_head_attention/query/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Omodel/transformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOpReadVariableOpVmodel_transformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
@model/transformer_block/multi_head_attention/key/einsum_1/EinsumEinsumAmodel/transformer_block/layer_normalization_1/batchnorm/add_1:z:0Wmodel/transformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Emodel/transformer_block/multi_head_attention/key/add_1/ReadVariableOpReadVariableOpLmodel_transformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
6model/transformer_block/multi_head_attention/key/add_1AddV2Imodel/transformer_block/multi_head_attention/key/einsum_1/Einsum:output:0Mmodel/transformer_block/multi_head_attention/key/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Qmodel/transformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOpReadVariableOpXmodel_transformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Bmodel/transformer_block/multi_head_attention/value/einsum_1/EinsumEinsumAmodel/transformer_block/layer_normalization_1/batchnorm/add_1:z:0Ymodel/transformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Gmodel/transformer_block/multi_head_attention/value/add_1/ReadVariableOpReadVariableOpNmodel_transformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
8model/transformer_block/multi_head_attention/value/add_1AddV2Kmodel/transformer_block/multi_head_attention/value/einsum_1/Einsum:output:0Omodel/transformer_block/multi_head_attention/value/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�y
4model/transformer_block/multi_head_attention/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
2model/transformer_block/multi_head_attention/Mul_1Mul<model/transformer_block/multi_head_attention/query/add_1:z:0=model/transformer_block/multi_head_attention/Mul_1/y:output:0*
T0*0
_output_shapes
:���������-��
<model/transformer_block/multi_head_attention/einsum_2/EinsumEinsum:model/transformer_block/multi_head_attention/key/add_1:z:06model/transformer_block/multi_head_attention/Mul_1:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
>model/transformer_block/multi_head_attention/softmax/Softmax_1SoftmaxEmodel/transformer_block/multi_head_attention/einsum_2/Einsum:output:0*
T0*/
_output_shapes
:���������--�
?model/transformer_block/multi_head_attention/dropout/Identity_1IdentityHmodel/transformer_block/multi_head_attention/softmax/Softmax_1:softmax:0*
T0*/
_output_shapes
:���������--�
<model/transformer_block/multi_head_attention/einsum_3/EinsumEinsumHmodel/transformer_block/multi_head_attention/dropout/Identity_1:output:0<model/transformer_block/multi_head_attention/value/add_1:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
\model/transformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOpReadVariableOpcmodel_transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Mmodel/transformer_block/multi_head_attention/attention_output/einsum_1/EinsumEinsumEmodel/transformer_block/multi_head_attention/einsum_3/Einsum:output:0dmodel/transformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
Rmodel/transformer_block/multi_head_attention/attention_output/add_1/ReadVariableOpReadVariableOpYmodel_transformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Cmodel/transformer_block/multi_head_attention/attention_output/add_1AddV2Vmodel/transformer_block/multi_head_attention/attention_output/einsum_1/Einsum:output:0Zmodel/transformer_block/multi_head_attention/attention_output/add_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
*model/transformer_block/dropout/Identity_1IdentityGmodel/transformer_block/multi_head_attention/attention_output/add_1:z:0*
T0*,
_output_shapes
:���������-��
model/transformer_block/add_2AddV2Amodel/transformer_block/layer_normalization_1/batchnorm/add_1:z:03model/transformer_block/dropout/Identity_1:output:0*
T0*,
_output_shapes
:���������-��
Lmodel/transformer_block/layer_normalization/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:model/transformer_block/layer_normalization/moments_1/meanMean!model/transformer_block/add_2:z:0Umodel/transformer_block/layer_normalization/moments_1/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
Bmodel/transformer_block/layer_normalization/moments_1/StopGradientStopGradientCmodel/transformer_block/layer_normalization/moments_1/mean:output:0*
T0*+
_output_shapes
:���������-�
Gmodel/transformer_block/layer_normalization/moments_1/SquaredDifferenceSquaredDifference!model/transformer_block/add_2:z:0Kmodel/transformer_block/layer_normalization/moments_1/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Pmodel/transformer_block/layer_normalization/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
>model/transformer_block/layer_normalization/moments_1/varianceMeanKmodel/transformer_block/layer_normalization/moments_1/SquaredDifference:z:0Ymodel/transformer_block/layer_normalization/moments_1/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
=model/transformer_block/layer_normalization/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
;model/transformer_block/layer_normalization/batchnorm_1/addAddV2Gmodel/transformer_block/layer_normalization/moments_1/variance:output:0Fmodel/transformer_block/layer_normalization/batchnorm_1/add/y:output:0*
T0*+
_output_shapes
:���������-�
=model/transformer_block/layer_normalization/batchnorm_1/RsqrtRsqrt?model/transformer_block/layer_normalization/batchnorm_1/add:z:0*
T0*+
_output_shapes
:���������-�
Jmodel/transformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpQmodel_transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;model/transformer_block/layer_normalization/batchnorm_1/mulMulAmodel/transformer_block/layer_normalization/batchnorm_1/Rsqrt:y:0Rmodel/transformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
=model/transformer_block/layer_normalization/batchnorm_1/mul_1Mul!model/transformer_block/add_2:z:0?model/transformer_block/layer_normalization/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:���������-��
=model/transformer_block/layer_normalization/batchnorm_1/mul_2MulCmodel/transformer_block/layer_normalization/moments_1/mean:output:0?model/transformer_block/layer_normalization/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:���������-��
Fmodel/transformer_block/layer_normalization/batchnorm_1/ReadVariableOpReadVariableOpMmodel_transformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;model/transformer_block/layer_normalization/batchnorm_1/subSubNmodel/transformer_block/layer_normalization/batchnorm_1/ReadVariableOp:value:0Amodel/transformer_block/layer_normalization/batchnorm_1/mul_2:z:0*
T0*,
_output_shapes
:���������-��
=model/transformer_block/layer_normalization/batchnorm_1/add_1AddV2Amodel/transformer_block/layer_normalization/batchnorm_1/mul_1:z:0?model/transformer_block/layer_normalization/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:���������-��
Cmodel/transformer_block/sequential/dense/Tensordot_1/ReadVariableOpReadVariableOpJmodel_transformer_block_sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
9model/transformer_block/sequential/dense/Tensordot_1/axesConst*
_output_shapes
:*
dtype0*
valueB:�
9model/transformer_block/sequential/dense/Tensordot_1/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
:model/transformer_block/sequential/dense/Tensordot_1/ShapeShapeAmodel/transformer_block/layer_normalization/batchnorm_1/add_1:z:0*
T0*
_output_shapes
:�
Bmodel/transformer_block/sequential/dense/Tensordot_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_block/sequential/dense/Tensordot_1/GatherV2GatherV2Cmodel/transformer_block/sequential/dense/Tensordot_1/Shape:output:0Bmodel/transformer_block/sequential/dense/Tensordot_1/free:output:0Kmodel/transformer_block/sequential/dense/Tensordot_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Dmodel/transformer_block/sequential/dense/Tensordot_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?model/transformer_block/sequential/dense/Tensordot_1/GatherV2_1GatherV2Cmodel/transformer_block/sequential/dense/Tensordot_1/Shape:output:0Bmodel/transformer_block/sequential/dense/Tensordot_1/axes:output:0Mmodel/transformer_block/sequential/dense/Tensordot_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
:model/transformer_block/sequential/dense/Tensordot_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
9model/transformer_block/sequential/dense/Tensordot_1/ProdProdFmodel/transformer_block/sequential/dense/Tensordot_1/GatherV2:output:0Cmodel/transformer_block/sequential/dense/Tensordot_1/Const:output:0*
T0*
_output_shapes
: �
<model/transformer_block/sequential/dense/Tensordot_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
;model/transformer_block/sequential/dense/Tensordot_1/Prod_1ProdHmodel/transformer_block/sequential/dense/Tensordot_1/GatherV2_1:output:0Emodel/transformer_block/sequential/dense/Tensordot_1/Const_1:output:0*
T0*
_output_shapes
: �
@model/transformer_block/sequential/dense/Tensordot_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;model/transformer_block/sequential/dense/Tensordot_1/concatConcatV2Bmodel/transformer_block/sequential/dense/Tensordot_1/free:output:0Bmodel/transformer_block/sequential/dense/Tensordot_1/axes:output:0Imodel/transformer_block/sequential/dense/Tensordot_1/concat/axis:output:0*
N*
T0*
_output_shapes
:�
:model/transformer_block/sequential/dense/Tensordot_1/stackPackBmodel/transformer_block/sequential/dense/Tensordot_1/Prod:output:0Dmodel/transformer_block/sequential/dense/Tensordot_1/Prod_1:output:0*
N*
T0*
_output_shapes
:�
>model/transformer_block/sequential/dense/Tensordot_1/transpose	TransposeAmodel/transformer_block/layer_normalization/batchnorm_1/add_1:z:0Dmodel/transformer_block/sequential/dense/Tensordot_1/concat:output:0*
T0*,
_output_shapes
:���������-��
<model/transformer_block/sequential/dense/Tensordot_1/ReshapeReshapeBmodel/transformer_block/sequential/dense/Tensordot_1/transpose:y:0Cmodel/transformer_block/sequential/dense/Tensordot_1/stack:output:0*
T0*0
_output_shapes
:�������������������
;model/transformer_block/sequential/dense/Tensordot_1/MatMulMatMulEmodel/transformer_block/sequential/dense/Tensordot_1/Reshape:output:0Kmodel/transformer_block/sequential/dense/Tensordot_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
<model/transformer_block/sequential/dense/Tensordot_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Bmodel/transformer_block/sequential/dense/Tensordot_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_block/sequential/dense/Tensordot_1/concat_1ConcatV2Fmodel/transformer_block/sequential/dense/Tensordot_1/GatherV2:output:0Emodel/transformer_block/sequential/dense/Tensordot_1/Const_2:output:0Kmodel/transformer_block/sequential/dense/Tensordot_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
4model/transformer_block/sequential/dense/Tensordot_1ReshapeEmodel/transformer_block/sequential/dense/Tensordot_1/MatMul:product:0Fmodel/transformer_block/sequential/dense/Tensordot_1/concat_1:output:0*
T0*,
_output_shapes
:���������-��
Amodel/transformer_block/sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOpHmodel_transformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
2model/transformer_block/sequential/dense/BiasAdd_1BiasAdd=model/transformer_block/sequential/dense/Tensordot_1:output:0Imodel/transformer_block/sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
/model/transformer_block/sequential/dense/Selu_1Selu;model/transformer_block/sequential/dense/BiasAdd_1:output:0*
T0*,
_output_shapes
:���������-��
Emodel/transformer_block/sequential/dense_1/Tensordot_1/ReadVariableOpReadVariableOpLmodel_transformer_block_sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
;model/transformer_block/sequential/dense_1/Tensordot_1/axesConst*
_output_shapes
:*
dtype0*
valueB:�
;model/transformer_block/sequential/dense_1/Tensordot_1/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
<model/transformer_block/sequential/dense_1/Tensordot_1/ShapeShape=model/transformer_block/sequential/dense/Selu_1:activations:0*
T0*
_output_shapes
:�
Dmodel/transformer_block/sequential/dense_1/Tensordot_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?model/transformer_block/sequential/dense_1/Tensordot_1/GatherV2GatherV2Emodel/transformer_block/sequential/dense_1/Tensordot_1/Shape:output:0Dmodel/transformer_block/sequential/dense_1/Tensordot_1/free:output:0Mmodel/transformer_block/sequential/dense_1/Tensordot_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Fmodel/transformer_block/sequential/dense_1/Tensordot_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Amodel/transformer_block/sequential/dense_1/Tensordot_1/GatherV2_1GatherV2Emodel/transformer_block/sequential/dense_1/Tensordot_1/Shape:output:0Dmodel/transformer_block/sequential/dense_1/Tensordot_1/axes:output:0Omodel/transformer_block/sequential/dense_1/Tensordot_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
<model/transformer_block/sequential/dense_1/Tensordot_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
;model/transformer_block/sequential/dense_1/Tensordot_1/ProdProdHmodel/transformer_block/sequential/dense_1/Tensordot_1/GatherV2:output:0Emodel/transformer_block/sequential/dense_1/Tensordot_1/Const:output:0*
T0*
_output_shapes
: �
>model/transformer_block/sequential/dense_1/Tensordot_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
=model/transformer_block/sequential/dense_1/Tensordot_1/Prod_1ProdJmodel/transformer_block/sequential/dense_1/Tensordot_1/GatherV2_1:output:0Gmodel/transformer_block/sequential/dense_1/Tensordot_1/Const_1:output:0*
T0*
_output_shapes
: �
Bmodel/transformer_block/sequential/dense_1/Tensordot_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_block/sequential/dense_1/Tensordot_1/concatConcatV2Dmodel/transformer_block/sequential/dense_1/Tensordot_1/free:output:0Dmodel/transformer_block/sequential/dense_1/Tensordot_1/axes:output:0Kmodel/transformer_block/sequential/dense_1/Tensordot_1/concat/axis:output:0*
N*
T0*
_output_shapes
:�
<model/transformer_block/sequential/dense_1/Tensordot_1/stackPackDmodel/transformer_block/sequential/dense_1/Tensordot_1/Prod:output:0Fmodel/transformer_block/sequential/dense_1/Tensordot_1/Prod_1:output:0*
N*
T0*
_output_shapes
:�
@model/transformer_block/sequential/dense_1/Tensordot_1/transpose	Transpose=model/transformer_block/sequential/dense/Selu_1:activations:0Fmodel/transformer_block/sequential/dense_1/Tensordot_1/concat:output:0*
T0*,
_output_shapes
:���������-��
>model/transformer_block/sequential/dense_1/Tensordot_1/ReshapeReshapeDmodel/transformer_block/sequential/dense_1/Tensordot_1/transpose:y:0Emodel/transformer_block/sequential/dense_1/Tensordot_1/stack:output:0*
T0*0
_output_shapes
:�������������������
=model/transformer_block/sequential/dense_1/Tensordot_1/MatMulMatMulGmodel/transformer_block/sequential/dense_1/Tensordot_1/Reshape:output:0Mmodel/transformer_block/sequential/dense_1/Tensordot_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
>model/transformer_block/sequential/dense_1/Tensordot_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Dmodel/transformer_block/sequential/dense_1/Tensordot_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?model/transformer_block/sequential/dense_1/Tensordot_1/concat_1ConcatV2Hmodel/transformer_block/sequential/dense_1/Tensordot_1/GatherV2:output:0Gmodel/transformer_block/sequential/dense_1/Tensordot_1/Const_2:output:0Mmodel/transformer_block/sequential/dense_1/Tensordot_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
6model/transformer_block/sequential/dense_1/Tensordot_1ReshapeGmodel/transformer_block/sequential/dense_1/Tensordot_1/MatMul:product:0Hmodel/transformer_block/sequential/dense_1/Tensordot_1/concat_1:output:0*
T0*,
_output_shapes
:���������-��
Cmodel/transformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOpReadVariableOpJmodel_transformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4model/transformer_block/sequential/dense_1/BiasAdd_1BiasAdd?model/transformer_block/sequential/dense_1/Tensordot_1:output:0Kmodel/transformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
,model/transformer_block/dropout_1/Identity_1Identity=model/transformer_block/sequential/dense_1/BiasAdd_1:output:0*
T0*,
_output_shapes
:���������-��
model/transformer_block/add_3AddV2Amodel/transformer_block/layer_normalization/batchnorm_1/add_1:z:05model/transformer_block/dropout_1/Identity_1:output:0*
T0*,
_output_shapes
:���������-��
Nmodel/transformer_block/layer_normalization_1/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
<model/transformer_block/layer_normalization_1/moments_1/meanMean!model/transformer_block/add_3:z:0Wmodel/transformer_block/layer_normalization_1/moments_1/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
Dmodel/transformer_block/layer_normalization_1/moments_1/StopGradientStopGradientEmodel/transformer_block/layer_normalization_1/moments_1/mean:output:0*
T0*+
_output_shapes
:���������-�
Imodel/transformer_block/layer_normalization_1/moments_1/SquaredDifferenceSquaredDifference!model/transformer_block/add_3:z:0Mmodel/transformer_block/layer_normalization_1/moments_1/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Rmodel/transformer_block/layer_normalization_1/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
@model/transformer_block/layer_normalization_1/moments_1/varianceMeanMmodel/transformer_block/layer_normalization_1/moments_1/SquaredDifference:z:0[model/transformer_block/layer_normalization_1/moments_1/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
?model/transformer_block/layer_normalization_1/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
=model/transformer_block/layer_normalization_1/batchnorm_1/addAddV2Imodel/transformer_block/layer_normalization_1/moments_1/variance:output:0Hmodel/transformer_block/layer_normalization_1/batchnorm_1/add/y:output:0*
T0*+
_output_shapes
:���������-�
?model/transformer_block/layer_normalization_1/batchnorm_1/RsqrtRsqrtAmodel/transformer_block/layer_normalization_1/batchnorm_1/add:z:0*
T0*+
_output_shapes
:���������-�
Lmodel/transformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOpReadVariableOpSmodel_transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=model/transformer_block/layer_normalization_1/batchnorm_1/mulMulCmodel/transformer_block/layer_normalization_1/batchnorm_1/Rsqrt:y:0Tmodel/transformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
?model/transformer_block/layer_normalization_1/batchnorm_1/mul_1Mul!model/transformer_block/add_3:z:0Amodel/transformer_block/layer_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:���������-��
?model/transformer_block/layer_normalization_1/batchnorm_1/mul_2MulEmodel/transformer_block/layer_normalization_1/moments_1/mean:output:0Amodel/transformer_block/layer_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:���������-��
Hmodel/transformer_block/layer_normalization_1/batchnorm_1/ReadVariableOpReadVariableOpOmodel_transformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=model/transformer_block/layer_normalization_1/batchnorm_1/subSubPmodel/transformer_block/layer_normalization_1/batchnorm_1/ReadVariableOp:value:0Cmodel/transformer_block/layer_normalization_1/batchnorm_1/mul_2:z:0*
T0*,
_output_shapes
:���������-��
?model/transformer_block/layer_normalization_1/batchnorm_1/add_1AddV2Cmodel/transformer_block/layer_normalization_1/batchnorm_1/mul_1:z:0Amodel/transformer_block/layer_normalization_1/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:���������-��
Qmodel/transformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOpReadVariableOpXmodel_transformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Bmodel/transformer_block/multi_head_attention/query/einsum_2/EinsumEinsumCmodel/transformer_block/layer_normalization_1/batchnorm_1/add_1:z:0Ymodel/transformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Gmodel/transformer_block/multi_head_attention/query/add_2/ReadVariableOpReadVariableOpNmodel_transformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
8model/transformer_block/multi_head_attention/query/add_2AddV2Kmodel/transformer_block/multi_head_attention/query/einsum_2/Einsum:output:0Omodel/transformer_block/multi_head_attention/query/add_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Omodel/transformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOpReadVariableOpVmodel_transformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
@model/transformer_block/multi_head_attention/key/einsum_2/EinsumEinsumCmodel/transformer_block/layer_normalization_1/batchnorm_1/add_1:z:0Wmodel/transformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Emodel/transformer_block/multi_head_attention/key/add_2/ReadVariableOpReadVariableOpLmodel_transformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
6model/transformer_block/multi_head_attention/key/add_2AddV2Imodel/transformer_block/multi_head_attention/key/einsum_2/Einsum:output:0Mmodel/transformer_block/multi_head_attention/key/add_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Qmodel/transformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOpReadVariableOpXmodel_transformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Bmodel/transformer_block/multi_head_attention/value/einsum_2/EinsumEinsumCmodel/transformer_block/layer_normalization_1/batchnorm_1/add_1:z:0Ymodel/transformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Gmodel/transformer_block/multi_head_attention/value/add_2/ReadVariableOpReadVariableOpNmodel_transformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
8model/transformer_block/multi_head_attention/value/add_2AddV2Kmodel/transformer_block/multi_head_attention/value/einsum_2/Einsum:output:0Omodel/transformer_block/multi_head_attention/value/add_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�y
4model/transformer_block/multi_head_attention/Mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
2model/transformer_block/multi_head_attention/Mul_2Mul<model/transformer_block/multi_head_attention/query/add_2:z:0=model/transformer_block/multi_head_attention/Mul_2/y:output:0*
T0*0
_output_shapes
:���������-��
<model/transformer_block/multi_head_attention/einsum_4/EinsumEinsum:model/transformer_block/multi_head_attention/key/add_2:z:06model/transformer_block/multi_head_attention/Mul_2:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
>model/transformer_block/multi_head_attention/softmax/Softmax_2SoftmaxEmodel/transformer_block/multi_head_attention/einsum_4/Einsum:output:0*
T0*/
_output_shapes
:���������--�
?model/transformer_block/multi_head_attention/dropout/Identity_2IdentityHmodel/transformer_block/multi_head_attention/softmax/Softmax_2:softmax:0*
T0*/
_output_shapes
:���������--�
<model/transformer_block/multi_head_attention/einsum_5/EinsumEinsumHmodel/transformer_block/multi_head_attention/dropout/Identity_2:output:0<model/transformer_block/multi_head_attention/value/add_2:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
\model/transformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOpReadVariableOpcmodel_transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Mmodel/transformer_block/multi_head_attention/attention_output/einsum_2/EinsumEinsumEmodel/transformer_block/multi_head_attention/einsum_5/Einsum:output:0dmodel/transformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
Rmodel/transformer_block/multi_head_attention/attention_output/add_2/ReadVariableOpReadVariableOpYmodel_transformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Cmodel/transformer_block/multi_head_attention/attention_output/add_2AddV2Vmodel/transformer_block/multi_head_attention/attention_output/einsum_2/Einsum:output:0Zmodel/transformer_block/multi_head_attention/attention_output/add_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
*model/transformer_block/dropout/Identity_2IdentityGmodel/transformer_block/multi_head_attention/attention_output/add_2:z:0*
T0*,
_output_shapes
:���������-��
model/transformer_block/add_4AddV2Cmodel/transformer_block/layer_normalization_1/batchnorm_1/add_1:z:03model/transformer_block/dropout/Identity_2:output:0*
T0*,
_output_shapes
:���������-��
Lmodel/transformer_block/layer_normalization/moments_2/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:model/transformer_block/layer_normalization/moments_2/meanMean!model/transformer_block/add_4:z:0Umodel/transformer_block/layer_normalization/moments_2/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
Bmodel/transformer_block/layer_normalization/moments_2/StopGradientStopGradientCmodel/transformer_block/layer_normalization/moments_2/mean:output:0*
T0*+
_output_shapes
:���������-�
Gmodel/transformer_block/layer_normalization/moments_2/SquaredDifferenceSquaredDifference!model/transformer_block/add_4:z:0Kmodel/transformer_block/layer_normalization/moments_2/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Pmodel/transformer_block/layer_normalization/moments_2/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
>model/transformer_block/layer_normalization/moments_2/varianceMeanKmodel/transformer_block/layer_normalization/moments_2/SquaredDifference:z:0Ymodel/transformer_block/layer_normalization/moments_2/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
=model/transformer_block/layer_normalization/batchnorm_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
;model/transformer_block/layer_normalization/batchnorm_2/addAddV2Gmodel/transformer_block/layer_normalization/moments_2/variance:output:0Fmodel/transformer_block/layer_normalization/batchnorm_2/add/y:output:0*
T0*+
_output_shapes
:���������-�
=model/transformer_block/layer_normalization/batchnorm_2/RsqrtRsqrt?model/transformer_block/layer_normalization/batchnorm_2/add:z:0*
T0*+
_output_shapes
:���������-�
Jmodel/transformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOpReadVariableOpQmodel_transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;model/transformer_block/layer_normalization/batchnorm_2/mulMulAmodel/transformer_block/layer_normalization/batchnorm_2/Rsqrt:y:0Rmodel/transformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
=model/transformer_block/layer_normalization/batchnorm_2/mul_1Mul!model/transformer_block/add_4:z:0?model/transformer_block/layer_normalization/batchnorm_2/mul:z:0*
T0*,
_output_shapes
:���������-��
=model/transformer_block/layer_normalization/batchnorm_2/mul_2MulCmodel/transformer_block/layer_normalization/moments_2/mean:output:0?model/transformer_block/layer_normalization/batchnorm_2/mul:z:0*
T0*,
_output_shapes
:���������-��
Fmodel/transformer_block/layer_normalization/batchnorm_2/ReadVariableOpReadVariableOpMmodel_transformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;model/transformer_block/layer_normalization/batchnorm_2/subSubNmodel/transformer_block/layer_normalization/batchnorm_2/ReadVariableOp:value:0Amodel/transformer_block/layer_normalization/batchnorm_2/mul_2:z:0*
T0*,
_output_shapes
:���������-��
=model/transformer_block/layer_normalization/batchnorm_2/add_1AddV2Amodel/transformer_block/layer_normalization/batchnorm_2/mul_1:z:0?model/transformer_block/layer_normalization/batchnorm_2/sub:z:0*
T0*,
_output_shapes
:���������-��
Cmodel/transformer_block/sequential/dense/Tensordot_2/ReadVariableOpReadVariableOpJmodel_transformer_block_sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
9model/transformer_block/sequential/dense/Tensordot_2/axesConst*
_output_shapes
:*
dtype0*
valueB:�
9model/transformer_block/sequential/dense/Tensordot_2/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
:model/transformer_block/sequential/dense/Tensordot_2/ShapeShapeAmodel/transformer_block/layer_normalization/batchnorm_2/add_1:z:0*
T0*
_output_shapes
:�
Bmodel/transformer_block/sequential/dense/Tensordot_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_block/sequential/dense/Tensordot_2/GatherV2GatherV2Cmodel/transformer_block/sequential/dense/Tensordot_2/Shape:output:0Bmodel/transformer_block/sequential/dense/Tensordot_2/free:output:0Kmodel/transformer_block/sequential/dense/Tensordot_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Dmodel/transformer_block/sequential/dense/Tensordot_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?model/transformer_block/sequential/dense/Tensordot_2/GatherV2_1GatherV2Cmodel/transformer_block/sequential/dense/Tensordot_2/Shape:output:0Bmodel/transformer_block/sequential/dense/Tensordot_2/axes:output:0Mmodel/transformer_block/sequential/dense/Tensordot_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
:model/transformer_block/sequential/dense/Tensordot_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
9model/transformer_block/sequential/dense/Tensordot_2/ProdProdFmodel/transformer_block/sequential/dense/Tensordot_2/GatherV2:output:0Cmodel/transformer_block/sequential/dense/Tensordot_2/Const:output:0*
T0*
_output_shapes
: �
<model/transformer_block/sequential/dense/Tensordot_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
;model/transformer_block/sequential/dense/Tensordot_2/Prod_1ProdHmodel/transformer_block/sequential/dense/Tensordot_2/GatherV2_1:output:0Emodel/transformer_block/sequential/dense/Tensordot_2/Const_1:output:0*
T0*
_output_shapes
: �
@model/transformer_block/sequential/dense/Tensordot_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;model/transformer_block/sequential/dense/Tensordot_2/concatConcatV2Bmodel/transformer_block/sequential/dense/Tensordot_2/free:output:0Bmodel/transformer_block/sequential/dense/Tensordot_2/axes:output:0Imodel/transformer_block/sequential/dense/Tensordot_2/concat/axis:output:0*
N*
T0*
_output_shapes
:�
:model/transformer_block/sequential/dense/Tensordot_2/stackPackBmodel/transformer_block/sequential/dense/Tensordot_2/Prod:output:0Dmodel/transformer_block/sequential/dense/Tensordot_2/Prod_1:output:0*
N*
T0*
_output_shapes
:�
>model/transformer_block/sequential/dense/Tensordot_2/transpose	TransposeAmodel/transformer_block/layer_normalization/batchnorm_2/add_1:z:0Dmodel/transformer_block/sequential/dense/Tensordot_2/concat:output:0*
T0*,
_output_shapes
:���������-��
<model/transformer_block/sequential/dense/Tensordot_2/ReshapeReshapeBmodel/transformer_block/sequential/dense/Tensordot_2/transpose:y:0Cmodel/transformer_block/sequential/dense/Tensordot_2/stack:output:0*
T0*0
_output_shapes
:�������������������
;model/transformer_block/sequential/dense/Tensordot_2/MatMulMatMulEmodel/transformer_block/sequential/dense/Tensordot_2/Reshape:output:0Kmodel/transformer_block/sequential/dense/Tensordot_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
<model/transformer_block/sequential/dense/Tensordot_2/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Bmodel/transformer_block/sequential/dense/Tensordot_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_block/sequential/dense/Tensordot_2/concat_1ConcatV2Fmodel/transformer_block/sequential/dense/Tensordot_2/GatherV2:output:0Emodel/transformer_block/sequential/dense/Tensordot_2/Const_2:output:0Kmodel/transformer_block/sequential/dense/Tensordot_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
4model/transformer_block/sequential/dense/Tensordot_2ReshapeEmodel/transformer_block/sequential/dense/Tensordot_2/MatMul:product:0Fmodel/transformer_block/sequential/dense/Tensordot_2/concat_1:output:0*
T0*,
_output_shapes
:���������-��
Amodel/transformer_block/sequential/dense/BiasAdd_2/ReadVariableOpReadVariableOpHmodel_transformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
2model/transformer_block/sequential/dense/BiasAdd_2BiasAdd=model/transformer_block/sequential/dense/Tensordot_2:output:0Imodel/transformer_block/sequential/dense/BiasAdd_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
/model/transformer_block/sequential/dense/Selu_2Selu;model/transformer_block/sequential/dense/BiasAdd_2:output:0*
T0*,
_output_shapes
:���������-��
Emodel/transformer_block/sequential/dense_1/Tensordot_2/ReadVariableOpReadVariableOpLmodel_transformer_block_sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
;model/transformer_block/sequential/dense_1/Tensordot_2/axesConst*
_output_shapes
:*
dtype0*
valueB:�
;model/transformer_block/sequential/dense_1/Tensordot_2/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
<model/transformer_block/sequential/dense_1/Tensordot_2/ShapeShape=model/transformer_block/sequential/dense/Selu_2:activations:0*
T0*
_output_shapes
:�
Dmodel/transformer_block/sequential/dense_1/Tensordot_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?model/transformer_block/sequential/dense_1/Tensordot_2/GatherV2GatherV2Emodel/transformer_block/sequential/dense_1/Tensordot_2/Shape:output:0Dmodel/transformer_block/sequential/dense_1/Tensordot_2/free:output:0Mmodel/transformer_block/sequential/dense_1/Tensordot_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Fmodel/transformer_block/sequential/dense_1/Tensordot_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Amodel/transformer_block/sequential/dense_1/Tensordot_2/GatherV2_1GatherV2Emodel/transformer_block/sequential/dense_1/Tensordot_2/Shape:output:0Dmodel/transformer_block/sequential/dense_1/Tensordot_2/axes:output:0Omodel/transformer_block/sequential/dense_1/Tensordot_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
<model/transformer_block/sequential/dense_1/Tensordot_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
;model/transformer_block/sequential/dense_1/Tensordot_2/ProdProdHmodel/transformer_block/sequential/dense_1/Tensordot_2/GatherV2:output:0Emodel/transformer_block/sequential/dense_1/Tensordot_2/Const:output:0*
T0*
_output_shapes
: �
>model/transformer_block/sequential/dense_1/Tensordot_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
=model/transformer_block/sequential/dense_1/Tensordot_2/Prod_1ProdJmodel/transformer_block/sequential/dense_1/Tensordot_2/GatherV2_1:output:0Gmodel/transformer_block/sequential/dense_1/Tensordot_2/Const_1:output:0*
T0*
_output_shapes
: �
Bmodel/transformer_block/sequential/dense_1/Tensordot_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_block/sequential/dense_1/Tensordot_2/concatConcatV2Dmodel/transformer_block/sequential/dense_1/Tensordot_2/free:output:0Dmodel/transformer_block/sequential/dense_1/Tensordot_2/axes:output:0Kmodel/transformer_block/sequential/dense_1/Tensordot_2/concat/axis:output:0*
N*
T0*
_output_shapes
:�
<model/transformer_block/sequential/dense_1/Tensordot_2/stackPackDmodel/transformer_block/sequential/dense_1/Tensordot_2/Prod:output:0Fmodel/transformer_block/sequential/dense_1/Tensordot_2/Prod_1:output:0*
N*
T0*
_output_shapes
:�
@model/transformer_block/sequential/dense_1/Tensordot_2/transpose	Transpose=model/transformer_block/sequential/dense/Selu_2:activations:0Fmodel/transformer_block/sequential/dense_1/Tensordot_2/concat:output:0*
T0*,
_output_shapes
:���������-��
>model/transformer_block/sequential/dense_1/Tensordot_2/ReshapeReshapeDmodel/transformer_block/sequential/dense_1/Tensordot_2/transpose:y:0Emodel/transformer_block/sequential/dense_1/Tensordot_2/stack:output:0*
T0*0
_output_shapes
:�������������������
=model/transformer_block/sequential/dense_1/Tensordot_2/MatMulMatMulGmodel/transformer_block/sequential/dense_1/Tensordot_2/Reshape:output:0Mmodel/transformer_block/sequential/dense_1/Tensordot_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
>model/transformer_block/sequential/dense_1/Tensordot_2/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Dmodel/transformer_block/sequential/dense_1/Tensordot_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?model/transformer_block/sequential/dense_1/Tensordot_2/concat_1ConcatV2Hmodel/transformer_block/sequential/dense_1/Tensordot_2/GatherV2:output:0Gmodel/transformer_block/sequential/dense_1/Tensordot_2/Const_2:output:0Mmodel/transformer_block/sequential/dense_1/Tensordot_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
6model/transformer_block/sequential/dense_1/Tensordot_2ReshapeGmodel/transformer_block/sequential/dense_1/Tensordot_2/MatMul:product:0Hmodel/transformer_block/sequential/dense_1/Tensordot_2/concat_1:output:0*
T0*,
_output_shapes
:���������-��
Cmodel/transformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOpReadVariableOpJmodel_transformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4model/transformer_block/sequential/dense_1/BiasAdd_2BiasAdd?model/transformer_block/sequential/dense_1/Tensordot_2:output:0Kmodel/transformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
,model/transformer_block/dropout_1/Identity_2Identity=model/transformer_block/sequential/dense_1/BiasAdd_2:output:0*
T0*,
_output_shapes
:���������-��
model/transformer_block/add_5AddV2Amodel/transformer_block/layer_normalization/batchnorm_2/add_1:z:05model/transformer_block/dropout_1/Identity_2:output:0*
T0*,
_output_shapes
:���������-��
Nmodel/transformer_block/layer_normalization_1/moments_2/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
<model/transformer_block/layer_normalization_1/moments_2/meanMean!model/transformer_block/add_5:z:0Wmodel/transformer_block/layer_normalization_1/moments_2/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
Dmodel/transformer_block/layer_normalization_1/moments_2/StopGradientStopGradientEmodel/transformer_block/layer_normalization_1/moments_2/mean:output:0*
T0*+
_output_shapes
:���������-�
Imodel/transformer_block/layer_normalization_1/moments_2/SquaredDifferenceSquaredDifference!model/transformer_block/add_5:z:0Mmodel/transformer_block/layer_normalization_1/moments_2/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Rmodel/transformer_block/layer_normalization_1/moments_2/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
@model/transformer_block/layer_normalization_1/moments_2/varianceMeanMmodel/transformer_block/layer_normalization_1/moments_2/SquaredDifference:z:0[model/transformer_block/layer_normalization_1/moments_2/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
?model/transformer_block/layer_normalization_1/batchnorm_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
=model/transformer_block/layer_normalization_1/batchnorm_2/addAddV2Imodel/transformer_block/layer_normalization_1/moments_2/variance:output:0Hmodel/transformer_block/layer_normalization_1/batchnorm_2/add/y:output:0*
T0*+
_output_shapes
:���������-�
?model/transformer_block/layer_normalization_1/batchnorm_2/RsqrtRsqrtAmodel/transformer_block/layer_normalization_1/batchnorm_2/add:z:0*
T0*+
_output_shapes
:���������-�
Lmodel/transformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOpReadVariableOpSmodel_transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=model/transformer_block/layer_normalization_1/batchnorm_2/mulMulCmodel/transformer_block/layer_normalization_1/batchnorm_2/Rsqrt:y:0Tmodel/transformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
?model/transformer_block/layer_normalization_1/batchnorm_2/mul_1Mul!model/transformer_block/add_5:z:0Amodel/transformer_block/layer_normalization_1/batchnorm_2/mul:z:0*
T0*,
_output_shapes
:���������-��
?model/transformer_block/layer_normalization_1/batchnorm_2/mul_2MulEmodel/transformer_block/layer_normalization_1/moments_2/mean:output:0Amodel/transformer_block/layer_normalization_1/batchnorm_2/mul:z:0*
T0*,
_output_shapes
:���������-��
Hmodel/transformer_block/layer_normalization_1/batchnorm_2/ReadVariableOpReadVariableOpOmodel_transformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=model/transformer_block/layer_normalization_1/batchnorm_2/subSubPmodel/transformer_block/layer_normalization_1/batchnorm_2/ReadVariableOp:value:0Cmodel/transformer_block/layer_normalization_1/batchnorm_2/mul_2:z:0*
T0*,
_output_shapes
:���������-��
?model/transformer_block/layer_normalization_1/batchnorm_2/add_1AddV2Cmodel/transformer_block/layer_normalization_1/batchnorm_2/mul_1:z:0Amodel/transformer_block/layer_normalization_1/batchnorm_2/sub:z:0*
T0*,
_output_shapes
:���������-��
Qmodel/transformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOpReadVariableOpXmodel_transformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Bmodel/transformer_block/multi_head_attention/query/einsum_3/EinsumEinsumCmodel/transformer_block/layer_normalization_1/batchnorm_2/add_1:z:0Ymodel/transformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Gmodel/transformer_block/multi_head_attention/query/add_3/ReadVariableOpReadVariableOpNmodel_transformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
8model/transformer_block/multi_head_attention/query/add_3AddV2Kmodel/transformer_block/multi_head_attention/query/einsum_3/Einsum:output:0Omodel/transformer_block/multi_head_attention/query/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Omodel/transformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOpReadVariableOpVmodel_transformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
@model/transformer_block/multi_head_attention/key/einsum_3/EinsumEinsumCmodel/transformer_block/layer_normalization_1/batchnorm_2/add_1:z:0Wmodel/transformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Emodel/transformer_block/multi_head_attention/key/add_3/ReadVariableOpReadVariableOpLmodel_transformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
6model/transformer_block/multi_head_attention/key/add_3AddV2Imodel/transformer_block/multi_head_attention/key/einsum_3/Einsum:output:0Mmodel/transformer_block/multi_head_attention/key/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Qmodel/transformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOpReadVariableOpXmodel_transformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Bmodel/transformer_block/multi_head_attention/value/einsum_3/EinsumEinsumCmodel/transformer_block/layer_normalization_1/batchnorm_2/add_1:z:0Ymodel/transformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Gmodel/transformer_block/multi_head_attention/value/add_3/ReadVariableOpReadVariableOpNmodel_transformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
8model/transformer_block/multi_head_attention/value/add_3AddV2Kmodel/transformer_block/multi_head_attention/value/einsum_3/Einsum:output:0Omodel/transformer_block/multi_head_attention/value/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�y
4model/transformer_block/multi_head_attention/Mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
2model/transformer_block/multi_head_attention/Mul_3Mul<model/transformer_block/multi_head_attention/query/add_3:z:0=model/transformer_block/multi_head_attention/Mul_3/y:output:0*
T0*0
_output_shapes
:���������-��
<model/transformer_block/multi_head_attention/einsum_6/EinsumEinsum:model/transformer_block/multi_head_attention/key/add_3:z:06model/transformer_block/multi_head_attention/Mul_3:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
>model/transformer_block/multi_head_attention/softmax/Softmax_3SoftmaxEmodel/transformer_block/multi_head_attention/einsum_6/Einsum:output:0*
T0*/
_output_shapes
:���������--�
?model/transformer_block/multi_head_attention/dropout/Identity_3IdentityHmodel/transformer_block/multi_head_attention/softmax/Softmax_3:softmax:0*
T0*/
_output_shapes
:���������--�
<model/transformer_block/multi_head_attention/einsum_7/EinsumEinsumHmodel/transformer_block/multi_head_attention/dropout/Identity_3:output:0<model/transformer_block/multi_head_attention/value/add_3:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
\model/transformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOpReadVariableOpcmodel_transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Mmodel/transformer_block/multi_head_attention/attention_output/einsum_3/EinsumEinsumEmodel/transformer_block/multi_head_attention/einsum_7/Einsum:output:0dmodel/transformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
Rmodel/transformer_block/multi_head_attention/attention_output/add_3/ReadVariableOpReadVariableOpYmodel_transformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Cmodel/transformer_block/multi_head_attention/attention_output/add_3AddV2Vmodel/transformer_block/multi_head_attention/attention_output/einsum_3/Einsum:output:0Zmodel/transformer_block/multi_head_attention/attention_output/add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
*model/transformer_block/dropout/Identity_3IdentityGmodel/transformer_block/multi_head_attention/attention_output/add_3:z:0*
T0*,
_output_shapes
:���������-��
model/transformer_block/add_6AddV2Cmodel/transformer_block/layer_normalization_1/batchnorm_2/add_1:z:03model/transformer_block/dropout/Identity_3:output:0*
T0*,
_output_shapes
:���������-��
Lmodel/transformer_block/layer_normalization/moments_3/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:model/transformer_block/layer_normalization/moments_3/meanMean!model/transformer_block/add_6:z:0Umodel/transformer_block/layer_normalization/moments_3/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
Bmodel/transformer_block/layer_normalization/moments_3/StopGradientStopGradientCmodel/transformer_block/layer_normalization/moments_3/mean:output:0*
T0*+
_output_shapes
:���������-�
Gmodel/transformer_block/layer_normalization/moments_3/SquaredDifferenceSquaredDifference!model/transformer_block/add_6:z:0Kmodel/transformer_block/layer_normalization/moments_3/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Pmodel/transformer_block/layer_normalization/moments_3/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
>model/transformer_block/layer_normalization/moments_3/varianceMeanKmodel/transformer_block/layer_normalization/moments_3/SquaredDifference:z:0Ymodel/transformer_block/layer_normalization/moments_3/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
=model/transformer_block/layer_normalization/batchnorm_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
;model/transformer_block/layer_normalization/batchnorm_3/addAddV2Gmodel/transformer_block/layer_normalization/moments_3/variance:output:0Fmodel/transformer_block/layer_normalization/batchnorm_3/add/y:output:0*
T0*+
_output_shapes
:���������-�
=model/transformer_block/layer_normalization/batchnorm_3/RsqrtRsqrt?model/transformer_block/layer_normalization/batchnorm_3/add:z:0*
T0*+
_output_shapes
:���������-�
Jmodel/transformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOpReadVariableOpQmodel_transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;model/transformer_block/layer_normalization/batchnorm_3/mulMulAmodel/transformer_block/layer_normalization/batchnorm_3/Rsqrt:y:0Rmodel/transformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
=model/transformer_block/layer_normalization/batchnorm_3/mul_1Mul!model/transformer_block/add_6:z:0?model/transformer_block/layer_normalization/batchnorm_3/mul:z:0*
T0*,
_output_shapes
:���������-��
=model/transformer_block/layer_normalization/batchnorm_3/mul_2MulCmodel/transformer_block/layer_normalization/moments_3/mean:output:0?model/transformer_block/layer_normalization/batchnorm_3/mul:z:0*
T0*,
_output_shapes
:���������-��
Fmodel/transformer_block/layer_normalization/batchnorm_3/ReadVariableOpReadVariableOpMmodel_transformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;model/transformer_block/layer_normalization/batchnorm_3/subSubNmodel/transformer_block/layer_normalization/batchnorm_3/ReadVariableOp:value:0Amodel/transformer_block/layer_normalization/batchnorm_3/mul_2:z:0*
T0*,
_output_shapes
:���������-��
=model/transformer_block/layer_normalization/batchnorm_3/add_1AddV2Amodel/transformer_block/layer_normalization/batchnorm_3/mul_1:z:0?model/transformer_block/layer_normalization/batchnorm_3/sub:z:0*
T0*,
_output_shapes
:���������-��
Cmodel/transformer_block/sequential/dense/Tensordot_3/ReadVariableOpReadVariableOpJmodel_transformer_block_sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
9model/transformer_block/sequential/dense/Tensordot_3/axesConst*
_output_shapes
:*
dtype0*
valueB:�
9model/transformer_block/sequential/dense/Tensordot_3/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
:model/transformer_block/sequential/dense/Tensordot_3/ShapeShapeAmodel/transformer_block/layer_normalization/batchnorm_3/add_1:z:0*
T0*
_output_shapes
:�
Bmodel/transformer_block/sequential/dense/Tensordot_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_block/sequential/dense/Tensordot_3/GatherV2GatherV2Cmodel/transformer_block/sequential/dense/Tensordot_3/Shape:output:0Bmodel/transformer_block/sequential/dense/Tensordot_3/free:output:0Kmodel/transformer_block/sequential/dense/Tensordot_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Dmodel/transformer_block/sequential/dense/Tensordot_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?model/transformer_block/sequential/dense/Tensordot_3/GatherV2_1GatherV2Cmodel/transformer_block/sequential/dense/Tensordot_3/Shape:output:0Bmodel/transformer_block/sequential/dense/Tensordot_3/axes:output:0Mmodel/transformer_block/sequential/dense/Tensordot_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
:model/transformer_block/sequential/dense/Tensordot_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
9model/transformer_block/sequential/dense/Tensordot_3/ProdProdFmodel/transformer_block/sequential/dense/Tensordot_3/GatherV2:output:0Cmodel/transformer_block/sequential/dense/Tensordot_3/Const:output:0*
T0*
_output_shapes
: �
<model/transformer_block/sequential/dense/Tensordot_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
;model/transformer_block/sequential/dense/Tensordot_3/Prod_1ProdHmodel/transformer_block/sequential/dense/Tensordot_3/GatherV2_1:output:0Emodel/transformer_block/sequential/dense/Tensordot_3/Const_1:output:0*
T0*
_output_shapes
: �
@model/transformer_block/sequential/dense/Tensordot_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;model/transformer_block/sequential/dense/Tensordot_3/concatConcatV2Bmodel/transformer_block/sequential/dense/Tensordot_3/free:output:0Bmodel/transformer_block/sequential/dense/Tensordot_3/axes:output:0Imodel/transformer_block/sequential/dense/Tensordot_3/concat/axis:output:0*
N*
T0*
_output_shapes
:�
:model/transformer_block/sequential/dense/Tensordot_3/stackPackBmodel/transformer_block/sequential/dense/Tensordot_3/Prod:output:0Dmodel/transformer_block/sequential/dense/Tensordot_3/Prod_1:output:0*
N*
T0*
_output_shapes
:�
>model/transformer_block/sequential/dense/Tensordot_3/transpose	TransposeAmodel/transformer_block/layer_normalization/batchnorm_3/add_1:z:0Dmodel/transformer_block/sequential/dense/Tensordot_3/concat:output:0*
T0*,
_output_shapes
:���������-��
<model/transformer_block/sequential/dense/Tensordot_3/ReshapeReshapeBmodel/transformer_block/sequential/dense/Tensordot_3/transpose:y:0Cmodel/transformer_block/sequential/dense/Tensordot_3/stack:output:0*
T0*0
_output_shapes
:�������������������
;model/transformer_block/sequential/dense/Tensordot_3/MatMulMatMulEmodel/transformer_block/sequential/dense/Tensordot_3/Reshape:output:0Kmodel/transformer_block/sequential/dense/Tensordot_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
<model/transformer_block/sequential/dense/Tensordot_3/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Bmodel/transformer_block/sequential/dense/Tensordot_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_block/sequential/dense/Tensordot_3/concat_1ConcatV2Fmodel/transformer_block/sequential/dense/Tensordot_3/GatherV2:output:0Emodel/transformer_block/sequential/dense/Tensordot_3/Const_2:output:0Kmodel/transformer_block/sequential/dense/Tensordot_3/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
4model/transformer_block/sequential/dense/Tensordot_3ReshapeEmodel/transformer_block/sequential/dense/Tensordot_3/MatMul:product:0Fmodel/transformer_block/sequential/dense/Tensordot_3/concat_1:output:0*
T0*,
_output_shapes
:���������-��
Amodel/transformer_block/sequential/dense/BiasAdd_3/ReadVariableOpReadVariableOpHmodel_transformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
2model/transformer_block/sequential/dense/BiasAdd_3BiasAdd=model/transformer_block/sequential/dense/Tensordot_3:output:0Imodel/transformer_block/sequential/dense/BiasAdd_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
/model/transformer_block/sequential/dense/Selu_3Selu;model/transformer_block/sequential/dense/BiasAdd_3:output:0*
T0*,
_output_shapes
:���������-��
Emodel/transformer_block/sequential/dense_1/Tensordot_3/ReadVariableOpReadVariableOpLmodel_transformer_block_sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
;model/transformer_block/sequential/dense_1/Tensordot_3/axesConst*
_output_shapes
:*
dtype0*
valueB:�
;model/transformer_block/sequential/dense_1/Tensordot_3/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
<model/transformer_block/sequential/dense_1/Tensordot_3/ShapeShape=model/transformer_block/sequential/dense/Selu_3:activations:0*
T0*
_output_shapes
:�
Dmodel/transformer_block/sequential/dense_1/Tensordot_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?model/transformer_block/sequential/dense_1/Tensordot_3/GatherV2GatherV2Emodel/transformer_block/sequential/dense_1/Tensordot_3/Shape:output:0Dmodel/transformer_block/sequential/dense_1/Tensordot_3/free:output:0Mmodel/transformer_block/sequential/dense_1/Tensordot_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Fmodel/transformer_block/sequential/dense_1/Tensordot_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Amodel/transformer_block/sequential/dense_1/Tensordot_3/GatherV2_1GatherV2Emodel/transformer_block/sequential/dense_1/Tensordot_3/Shape:output:0Dmodel/transformer_block/sequential/dense_1/Tensordot_3/axes:output:0Omodel/transformer_block/sequential/dense_1/Tensordot_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
<model/transformer_block/sequential/dense_1/Tensordot_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
;model/transformer_block/sequential/dense_1/Tensordot_3/ProdProdHmodel/transformer_block/sequential/dense_1/Tensordot_3/GatherV2:output:0Emodel/transformer_block/sequential/dense_1/Tensordot_3/Const:output:0*
T0*
_output_shapes
: �
>model/transformer_block/sequential/dense_1/Tensordot_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
=model/transformer_block/sequential/dense_1/Tensordot_3/Prod_1ProdJmodel/transformer_block/sequential/dense_1/Tensordot_3/GatherV2_1:output:0Gmodel/transformer_block/sequential/dense_1/Tensordot_3/Const_1:output:0*
T0*
_output_shapes
: �
Bmodel/transformer_block/sequential/dense_1/Tensordot_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_block/sequential/dense_1/Tensordot_3/concatConcatV2Dmodel/transformer_block/sequential/dense_1/Tensordot_3/free:output:0Dmodel/transformer_block/sequential/dense_1/Tensordot_3/axes:output:0Kmodel/transformer_block/sequential/dense_1/Tensordot_3/concat/axis:output:0*
N*
T0*
_output_shapes
:�
<model/transformer_block/sequential/dense_1/Tensordot_3/stackPackDmodel/transformer_block/sequential/dense_1/Tensordot_3/Prod:output:0Fmodel/transformer_block/sequential/dense_1/Tensordot_3/Prod_1:output:0*
N*
T0*
_output_shapes
:�
@model/transformer_block/sequential/dense_1/Tensordot_3/transpose	Transpose=model/transformer_block/sequential/dense/Selu_3:activations:0Fmodel/transformer_block/sequential/dense_1/Tensordot_3/concat:output:0*
T0*,
_output_shapes
:���������-��
>model/transformer_block/sequential/dense_1/Tensordot_3/ReshapeReshapeDmodel/transformer_block/sequential/dense_1/Tensordot_3/transpose:y:0Emodel/transformer_block/sequential/dense_1/Tensordot_3/stack:output:0*
T0*0
_output_shapes
:�������������������
=model/transformer_block/sequential/dense_1/Tensordot_3/MatMulMatMulGmodel/transformer_block/sequential/dense_1/Tensordot_3/Reshape:output:0Mmodel/transformer_block/sequential/dense_1/Tensordot_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
>model/transformer_block/sequential/dense_1/Tensordot_3/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Dmodel/transformer_block/sequential/dense_1/Tensordot_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?model/transformer_block/sequential/dense_1/Tensordot_3/concat_1ConcatV2Hmodel/transformer_block/sequential/dense_1/Tensordot_3/GatherV2:output:0Gmodel/transformer_block/sequential/dense_1/Tensordot_3/Const_2:output:0Mmodel/transformer_block/sequential/dense_1/Tensordot_3/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
6model/transformer_block/sequential/dense_1/Tensordot_3ReshapeGmodel/transformer_block/sequential/dense_1/Tensordot_3/MatMul:product:0Hmodel/transformer_block/sequential/dense_1/Tensordot_3/concat_1:output:0*
T0*,
_output_shapes
:���������-��
Cmodel/transformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOpReadVariableOpJmodel_transformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4model/transformer_block/sequential/dense_1/BiasAdd_3BiasAdd?model/transformer_block/sequential/dense_1/Tensordot_3:output:0Kmodel/transformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
,model/transformer_block/dropout_1/Identity_3Identity=model/transformer_block/sequential/dense_1/BiasAdd_3:output:0*
T0*,
_output_shapes
:���������-��
model/transformer_block/add_7AddV2Amodel/transformer_block/layer_normalization/batchnorm_3/add_1:z:05model/transformer_block/dropout_1/Identity_3:output:0*
T0*,
_output_shapes
:���������-��
Nmodel/transformer_block/layer_normalization_1/moments_3/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
<model/transformer_block/layer_normalization_1/moments_3/meanMean!model/transformer_block/add_7:z:0Wmodel/transformer_block/layer_normalization_1/moments_3/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
Dmodel/transformer_block/layer_normalization_1/moments_3/StopGradientStopGradientEmodel/transformer_block/layer_normalization_1/moments_3/mean:output:0*
T0*+
_output_shapes
:���������-�
Imodel/transformer_block/layer_normalization_1/moments_3/SquaredDifferenceSquaredDifference!model/transformer_block/add_7:z:0Mmodel/transformer_block/layer_normalization_1/moments_3/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Rmodel/transformer_block/layer_normalization_1/moments_3/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
@model/transformer_block/layer_normalization_1/moments_3/varianceMeanMmodel/transformer_block/layer_normalization_1/moments_3/SquaredDifference:z:0[model/transformer_block/layer_normalization_1/moments_3/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
?model/transformer_block/layer_normalization_1/batchnorm_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
=model/transformer_block/layer_normalization_1/batchnorm_3/addAddV2Imodel/transformer_block/layer_normalization_1/moments_3/variance:output:0Hmodel/transformer_block/layer_normalization_1/batchnorm_3/add/y:output:0*
T0*+
_output_shapes
:���������-�
?model/transformer_block/layer_normalization_1/batchnorm_3/RsqrtRsqrtAmodel/transformer_block/layer_normalization_1/batchnorm_3/add:z:0*
T0*+
_output_shapes
:���������-�
Lmodel/transformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOpReadVariableOpSmodel_transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=model/transformer_block/layer_normalization_1/batchnorm_3/mulMulCmodel/transformer_block/layer_normalization_1/batchnorm_3/Rsqrt:y:0Tmodel/transformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
?model/transformer_block/layer_normalization_1/batchnorm_3/mul_1Mul!model/transformer_block/add_7:z:0Amodel/transformer_block/layer_normalization_1/batchnorm_3/mul:z:0*
T0*,
_output_shapes
:���������-��
?model/transformer_block/layer_normalization_1/batchnorm_3/mul_2MulEmodel/transformer_block/layer_normalization_1/moments_3/mean:output:0Amodel/transformer_block/layer_normalization_1/batchnorm_3/mul:z:0*
T0*,
_output_shapes
:���������-��
Hmodel/transformer_block/layer_normalization_1/batchnorm_3/ReadVariableOpReadVariableOpOmodel_transformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=model/transformer_block/layer_normalization_1/batchnorm_3/subSubPmodel/transformer_block/layer_normalization_1/batchnorm_3/ReadVariableOp:value:0Cmodel/transformer_block/layer_normalization_1/batchnorm_3/mul_2:z:0*
T0*,
_output_shapes
:���������-��
?model/transformer_block/layer_normalization_1/batchnorm_3/add_1AddV2Cmodel/transformer_block/layer_normalization_1/batchnorm_3/mul_1:z:0Amodel/transformer_block/layer_normalization_1/batchnorm_3/sub:z:0*
T0*,
_output_shapes
:���������-��
Qmodel/transformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOpReadVariableOpXmodel_transformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Bmodel/transformer_block/multi_head_attention/query/einsum_4/EinsumEinsumCmodel/transformer_block/layer_normalization_1/batchnorm_3/add_1:z:0Ymodel/transformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Gmodel/transformer_block/multi_head_attention/query/add_4/ReadVariableOpReadVariableOpNmodel_transformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
8model/transformer_block/multi_head_attention/query/add_4AddV2Kmodel/transformer_block/multi_head_attention/query/einsum_4/Einsum:output:0Omodel/transformer_block/multi_head_attention/query/add_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Omodel/transformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOpReadVariableOpVmodel_transformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
@model/transformer_block/multi_head_attention/key/einsum_4/EinsumEinsumCmodel/transformer_block/layer_normalization_1/batchnorm_3/add_1:z:0Wmodel/transformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Emodel/transformer_block/multi_head_attention/key/add_4/ReadVariableOpReadVariableOpLmodel_transformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
6model/transformer_block/multi_head_attention/key/add_4AddV2Imodel/transformer_block/multi_head_attention/key/einsum_4/Einsum:output:0Mmodel/transformer_block/multi_head_attention/key/add_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
Qmodel/transformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOpReadVariableOpXmodel_transformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Bmodel/transformer_block/multi_head_attention/value/einsum_4/EinsumEinsumCmodel/transformer_block/layer_normalization_1/batchnorm_3/add_1:z:0Ymodel/transformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
Gmodel/transformer_block/multi_head_attention/value/add_4/ReadVariableOpReadVariableOpNmodel_transformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
8model/transformer_block/multi_head_attention/value/add_4AddV2Kmodel/transformer_block/multi_head_attention/value/einsum_4/Einsum:output:0Omodel/transformer_block/multi_head_attention/value/add_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�y
4model/transformer_block/multi_head_attention/Mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
2model/transformer_block/multi_head_attention/Mul_4Mul<model/transformer_block/multi_head_attention/query/add_4:z:0=model/transformer_block/multi_head_attention/Mul_4/y:output:0*
T0*0
_output_shapes
:���������-��
<model/transformer_block/multi_head_attention/einsum_8/EinsumEinsum:model/transformer_block/multi_head_attention/key/add_4:z:06model/transformer_block/multi_head_attention/Mul_4:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
>model/transformer_block/multi_head_attention/softmax/Softmax_4SoftmaxEmodel/transformer_block/multi_head_attention/einsum_8/Einsum:output:0*
T0*/
_output_shapes
:���������--�
?model/transformer_block/multi_head_attention/dropout/Identity_4IdentityHmodel/transformer_block/multi_head_attention/softmax/Softmax_4:softmax:0*
T0*/
_output_shapes
:���������--�
<model/transformer_block/multi_head_attention/einsum_9/EinsumEinsumHmodel/transformer_block/multi_head_attention/dropout/Identity_4:output:0<model/transformer_block/multi_head_attention/value/add_4:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
\model/transformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOpReadVariableOpcmodel_transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
Mmodel/transformer_block/multi_head_attention/attention_output/einsum_4/EinsumEinsumEmodel/transformer_block/multi_head_attention/einsum_9/Einsum:output:0dmodel/transformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
Rmodel/transformer_block/multi_head_attention/attention_output/add_4/ReadVariableOpReadVariableOpYmodel_transformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Cmodel/transformer_block/multi_head_attention/attention_output/add_4AddV2Vmodel/transformer_block/multi_head_attention/attention_output/einsum_4/Einsum:output:0Zmodel/transformer_block/multi_head_attention/attention_output/add_4/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
*model/transformer_block/dropout/Identity_4IdentityGmodel/transformer_block/multi_head_attention/attention_output/add_4:z:0*
T0*,
_output_shapes
:���������-��
model/transformer_block/add_8AddV2Cmodel/transformer_block/layer_normalization_1/batchnorm_3/add_1:z:03model/transformer_block/dropout/Identity_4:output:0*
T0*,
_output_shapes
:���������-��
Lmodel/transformer_block/layer_normalization/moments_4/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:model/transformer_block/layer_normalization/moments_4/meanMean!model/transformer_block/add_8:z:0Umodel/transformer_block/layer_normalization/moments_4/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
Bmodel/transformer_block/layer_normalization/moments_4/StopGradientStopGradientCmodel/transformer_block/layer_normalization/moments_4/mean:output:0*
T0*+
_output_shapes
:���������-�
Gmodel/transformer_block/layer_normalization/moments_4/SquaredDifferenceSquaredDifference!model/transformer_block/add_8:z:0Kmodel/transformer_block/layer_normalization/moments_4/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Pmodel/transformer_block/layer_normalization/moments_4/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
>model/transformer_block/layer_normalization/moments_4/varianceMeanKmodel/transformer_block/layer_normalization/moments_4/SquaredDifference:z:0Ymodel/transformer_block/layer_normalization/moments_4/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
=model/transformer_block/layer_normalization/batchnorm_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
;model/transformer_block/layer_normalization/batchnorm_4/addAddV2Gmodel/transformer_block/layer_normalization/moments_4/variance:output:0Fmodel/transformer_block/layer_normalization/batchnorm_4/add/y:output:0*
T0*+
_output_shapes
:���������-�
=model/transformer_block/layer_normalization/batchnorm_4/RsqrtRsqrt?model/transformer_block/layer_normalization/batchnorm_4/add:z:0*
T0*+
_output_shapes
:���������-�
Jmodel/transformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOpReadVariableOpQmodel_transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;model/transformer_block/layer_normalization/batchnorm_4/mulMulAmodel/transformer_block/layer_normalization/batchnorm_4/Rsqrt:y:0Rmodel/transformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
=model/transformer_block/layer_normalization/batchnorm_4/mul_1Mul!model/transformer_block/add_8:z:0?model/transformer_block/layer_normalization/batchnorm_4/mul:z:0*
T0*,
_output_shapes
:���������-��
=model/transformer_block/layer_normalization/batchnorm_4/mul_2MulCmodel/transformer_block/layer_normalization/moments_4/mean:output:0?model/transformer_block/layer_normalization/batchnorm_4/mul:z:0*
T0*,
_output_shapes
:���������-��
Fmodel/transformer_block/layer_normalization/batchnorm_4/ReadVariableOpReadVariableOpMmodel_transformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;model/transformer_block/layer_normalization/batchnorm_4/subSubNmodel/transformer_block/layer_normalization/batchnorm_4/ReadVariableOp:value:0Amodel/transformer_block/layer_normalization/batchnorm_4/mul_2:z:0*
T0*,
_output_shapes
:���������-��
=model/transformer_block/layer_normalization/batchnorm_4/add_1AddV2Amodel/transformer_block/layer_normalization/batchnorm_4/mul_1:z:0?model/transformer_block/layer_normalization/batchnorm_4/sub:z:0*
T0*,
_output_shapes
:���������-��
Cmodel/transformer_block/sequential/dense/Tensordot_4/ReadVariableOpReadVariableOpJmodel_transformer_block_sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
9model/transformer_block/sequential/dense/Tensordot_4/axesConst*
_output_shapes
:*
dtype0*
valueB:�
9model/transformer_block/sequential/dense/Tensordot_4/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
:model/transformer_block/sequential/dense/Tensordot_4/ShapeShapeAmodel/transformer_block/layer_normalization/batchnorm_4/add_1:z:0*
T0*
_output_shapes
:�
Bmodel/transformer_block/sequential/dense/Tensordot_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_block/sequential/dense/Tensordot_4/GatherV2GatherV2Cmodel/transformer_block/sequential/dense/Tensordot_4/Shape:output:0Bmodel/transformer_block/sequential/dense/Tensordot_4/free:output:0Kmodel/transformer_block/sequential/dense/Tensordot_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Dmodel/transformer_block/sequential/dense/Tensordot_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?model/transformer_block/sequential/dense/Tensordot_4/GatherV2_1GatherV2Cmodel/transformer_block/sequential/dense/Tensordot_4/Shape:output:0Bmodel/transformer_block/sequential/dense/Tensordot_4/axes:output:0Mmodel/transformer_block/sequential/dense/Tensordot_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
:model/transformer_block/sequential/dense/Tensordot_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
9model/transformer_block/sequential/dense/Tensordot_4/ProdProdFmodel/transformer_block/sequential/dense/Tensordot_4/GatherV2:output:0Cmodel/transformer_block/sequential/dense/Tensordot_4/Const:output:0*
T0*
_output_shapes
: �
<model/transformer_block/sequential/dense/Tensordot_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
;model/transformer_block/sequential/dense/Tensordot_4/Prod_1ProdHmodel/transformer_block/sequential/dense/Tensordot_4/GatherV2_1:output:0Emodel/transformer_block/sequential/dense/Tensordot_4/Const_1:output:0*
T0*
_output_shapes
: �
@model/transformer_block/sequential/dense/Tensordot_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;model/transformer_block/sequential/dense/Tensordot_4/concatConcatV2Bmodel/transformer_block/sequential/dense/Tensordot_4/free:output:0Bmodel/transformer_block/sequential/dense/Tensordot_4/axes:output:0Imodel/transformer_block/sequential/dense/Tensordot_4/concat/axis:output:0*
N*
T0*
_output_shapes
:�
:model/transformer_block/sequential/dense/Tensordot_4/stackPackBmodel/transformer_block/sequential/dense/Tensordot_4/Prod:output:0Dmodel/transformer_block/sequential/dense/Tensordot_4/Prod_1:output:0*
N*
T0*
_output_shapes
:�
>model/transformer_block/sequential/dense/Tensordot_4/transpose	TransposeAmodel/transformer_block/layer_normalization/batchnorm_4/add_1:z:0Dmodel/transformer_block/sequential/dense/Tensordot_4/concat:output:0*
T0*,
_output_shapes
:���������-��
<model/transformer_block/sequential/dense/Tensordot_4/ReshapeReshapeBmodel/transformer_block/sequential/dense/Tensordot_4/transpose:y:0Cmodel/transformer_block/sequential/dense/Tensordot_4/stack:output:0*
T0*0
_output_shapes
:�������������������
;model/transformer_block/sequential/dense/Tensordot_4/MatMulMatMulEmodel/transformer_block/sequential/dense/Tensordot_4/Reshape:output:0Kmodel/transformer_block/sequential/dense/Tensordot_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
<model/transformer_block/sequential/dense/Tensordot_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Bmodel/transformer_block/sequential/dense/Tensordot_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_block/sequential/dense/Tensordot_4/concat_1ConcatV2Fmodel/transformer_block/sequential/dense/Tensordot_4/GatherV2:output:0Emodel/transformer_block/sequential/dense/Tensordot_4/Const_2:output:0Kmodel/transformer_block/sequential/dense/Tensordot_4/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
4model/transformer_block/sequential/dense/Tensordot_4ReshapeEmodel/transformer_block/sequential/dense/Tensordot_4/MatMul:product:0Fmodel/transformer_block/sequential/dense/Tensordot_4/concat_1:output:0*
T0*,
_output_shapes
:���������-��
Amodel/transformer_block/sequential/dense/BiasAdd_4/ReadVariableOpReadVariableOpHmodel_transformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
2model/transformer_block/sequential/dense/BiasAdd_4BiasAdd=model/transformer_block/sequential/dense/Tensordot_4:output:0Imodel/transformer_block/sequential/dense/BiasAdd_4/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
/model/transformer_block/sequential/dense/Selu_4Selu;model/transformer_block/sequential/dense/BiasAdd_4:output:0*
T0*,
_output_shapes
:���������-��
Emodel/transformer_block/sequential/dense_1/Tensordot_4/ReadVariableOpReadVariableOpLmodel_transformer_block_sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
;model/transformer_block/sequential/dense_1/Tensordot_4/axesConst*
_output_shapes
:*
dtype0*
valueB:�
;model/transformer_block/sequential/dense_1/Tensordot_4/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
<model/transformer_block/sequential/dense_1/Tensordot_4/ShapeShape=model/transformer_block/sequential/dense/Selu_4:activations:0*
T0*
_output_shapes
:�
Dmodel/transformer_block/sequential/dense_1/Tensordot_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?model/transformer_block/sequential/dense_1/Tensordot_4/GatherV2GatherV2Emodel/transformer_block/sequential/dense_1/Tensordot_4/Shape:output:0Dmodel/transformer_block/sequential/dense_1/Tensordot_4/free:output:0Mmodel/transformer_block/sequential/dense_1/Tensordot_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Fmodel/transformer_block/sequential/dense_1/Tensordot_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Amodel/transformer_block/sequential/dense_1/Tensordot_4/GatherV2_1GatherV2Emodel/transformer_block/sequential/dense_1/Tensordot_4/Shape:output:0Dmodel/transformer_block/sequential/dense_1/Tensordot_4/axes:output:0Omodel/transformer_block/sequential/dense_1/Tensordot_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
<model/transformer_block/sequential/dense_1/Tensordot_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
;model/transformer_block/sequential/dense_1/Tensordot_4/ProdProdHmodel/transformer_block/sequential/dense_1/Tensordot_4/GatherV2:output:0Emodel/transformer_block/sequential/dense_1/Tensordot_4/Const:output:0*
T0*
_output_shapes
: �
>model/transformer_block/sequential/dense_1/Tensordot_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
=model/transformer_block/sequential/dense_1/Tensordot_4/Prod_1ProdJmodel/transformer_block/sequential/dense_1/Tensordot_4/GatherV2_1:output:0Gmodel/transformer_block/sequential/dense_1/Tensordot_4/Const_1:output:0*
T0*
_output_shapes
: �
Bmodel/transformer_block/sequential/dense_1/Tensordot_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=model/transformer_block/sequential/dense_1/Tensordot_4/concatConcatV2Dmodel/transformer_block/sequential/dense_1/Tensordot_4/free:output:0Dmodel/transformer_block/sequential/dense_1/Tensordot_4/axes:output:0Kmodel/transformer_block/sequential/dense_1/Tensordot_4/concat/axis:output:0*
N*
T0*
_output_shapes
:�
<model/transformer_block/sequential/dense_1/Tensordot_4/stackPackDmodel/transformer_block/sequential/dense_1/Tensordot_4/Prod:output:0Fmodel/transformer_block/sequential/dense_1/Tensordot_4/Prod_1:output:0*
N*
T0*
_output_shapes
:�
@model/transformer_block/sequential/dense_1/Tensordot_4/transpose	Transpose=model/transformer_block/sequential/dense/Selu_4:activations:0Fmodel/transformer_block/sequential/dense_1/Tensordot_4/concat:output:0*
T0*,
_output_shapes
:���������-��
>model/transformer_block/sequential/dense_1/Tensordot_4/ReshapeReshapeDmodel/transformer_block/sequential/dense_1/Tensordot_4/transpose:y:0Emodel/transformer_block/sequential/dense_1/Tensordot_4/stack:output:0*
T0*0
_output_shapes
:�������������������
=model/transformer_block/sequential/dense_1/Tensordot_4/MatMulMatMulGmodel/transformer_block/sequential/dense_1/Tensordot_4/Reshape:output:0Mmodel/transformer_block/sequential/dense_1/Tensordot_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
>model/transformer_block/sequential/dense_1/Tensordot_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Dmodel/transformer_block/sequential/dense_1/Tensordot_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?model/transformer_block/sequential/dense_1/Tensordot_4/concat_1ConcatV2Hmodel/transformer_block/sequential/dense_1/Tensordot_4/GatherV2:output:0Gmodel/transformer_block/sequential/dense_1/Tensordot_4/Const_2:output:0Mmodel/transformer_block/sequential/dense_1/Tensordot_4/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
6model/transformer_block/sequential/dense_1/Tensordot_4ReshapeGmodel/transformer_block/sequential/dense_1/Tensordot_4/MatMul:product:0Hmodel/transformer_block/sequential/dense_1/Tensordot_4/concat_1:output:0*
T0*,
_output_shapes
:���������-��
Cmodel/transformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOpReadVariableOpJmodel_transformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4model/transformer_block/sequential/dense_1/BiasAdd_4BiasAdd?model/transformer_block/sequential/dense_1/Tensordot_4:output:0Kmodel/transformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
,model/transformer_block/dropout_1/Identity_4Identity=model/transformer_block/sequential/dense_1/BiasAdd_4:output:0*
T0*,
_output_shapes
:���������-��
model/transformer_block/add_9AddV2Amodel/transformer_block/layer_normalization/batchnorm_4/add_1:z:05model/transformer_block/dropout_1/Identity_4:output:0*
T0*,
_output_shapes
:���������-��
Nmodel/transformer_block/layer_normalization_1/moments_4/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
<model/transformer_block/layer_normalization_1/moments_4/meanMean!model/transformer_block/add_9:z:0Wmodel/transformer_block/layer_normalization_1/moments_4/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
Dmodel/transformer_block/layer_normalization_1/moments_4/StopGradientStopGradientEmodel/transformer_block/layer_normalization_1/moments_4/mean:output:0*
T0*+
_output_shapes
:���������-�
Imodel/transformer_block/layer_normalization_1/moments_4/SquaredDifferenceSquaredDifference!model/transformer_block/add_9:z:0Mmodel/transformer_block/layer_normalization_1/moments_4/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
Rmodel/transformer_block/layer_normalization_1/moments_4/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
@model/transformer_block/layer_normalization_1/moments_4/varianceMeanMmodel/transformer_block/layer_normalization_1/moments_4/SquaredDifference:z:0[model/transformer_block/layer_normalization_1/moments_4/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
?model/transformer_block/layer_normalization_1/batchnorm_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
=model/transformer_block/layer_normalization_1/batchnorm_4/addAddV2Imodel/transformer_block/layer_normalization_1/moments_4/variance:output:0Hmodel/transformer_block/layer_normalization_1/batchnorm_4/add/y:output:0*
T0*+
_output_shapes
:���������-�
?model/transformer_block/layer_normalization_1/batchnorm_4/RsqrtRsqrtAmodel/transformer_block/layer_normalization_1/batchnorm_4/add:z:0*
T0*+
_output_shapes
:���������-�
Lmodel/transformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOpReadVariableOpSmodel_transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=model/transformer_block/layer_normalization_1/batchnorm_4/mulMulCmodel/transformer_block/layer_normalization_1/batchnorm_4/Rsqrt:y:0Tmodel/transformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
?model/transformer_block/layer_normalization_1/batchnorm_4/mul_1Mul!model/transformer_block/add_9:z:0Amodel/transformer_block/layer_normalization_1/batchnorm_4/mul:z:0*
T0*,
_output_shapes
:���������-��
?model/transformer_block/layer_normalization_1/batchnorm_4/mul_2MulEmodel/transformer_block/layer_normalization_1/moments_4/mean:output:0Amodel/transformer_block/layer_normalization_1/batchnorm_4/mul:z:0*
T0*,
_output_shapes
:���������-��
Hmodel/transformer_block/layer_normalization_1/batchnorm_4/ReadVariableOpReadVariableOpOmodel_transformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=model/transformer_block/layer_normalization_1/batchnorm_4/subSubPmodel/transformer_block/layer_normalization_1/batchnorm_4/ReadVariableOp:value:0Cmodel/transformer_block/layer_normalization_1/batchnorm_4/mul_2:z:0*
T0*,
_output_shapes
:���������-��
?model/transformer_block/layer_normalization_1/batchnorm_4/add_1AddV2Cmodel/transformer_block/layer_normalization_1/batchnorm_4/mul_1:z:0Amodel/transformer_block/layer_normalization_1/batchnorm_4/sub:z:0*
T0*,
_output_shapes
:���������-��
model/reshape/ShapeShapeCmodel/transformer_block/layer_normalization_1/batchnorm_4/add_1:z:0*
T0*
_output_shapes
:k
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :`
model/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :�-�
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0&model/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
model/reshape/ReshapeReshapeCmodel/transformer_block/layer_normalization_1/batchnorm_4/add_1:z:0$model/reshape/Reshape/shape:output:0*
T0*,
_output_shapes
:����������-{
model/dropout_2/IdentityIdentitymodel/reshape/Reshape:output:0*
T0*,
_output_shapes
:����������-�
&model/dense_2/Tensordot/ReadVariableOpReadVariableOp/model_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
�-�*
dtype0f
model/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
model/dense_2/Tensordot/ShapeShape!model/dropout_2/Identity:output:0*
T0*
_output_shapes
:g
%model/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model/dense_2/Tensordot/GatherV2GatherV2&model/dense_2/Tensordot/Shape:output:0%model/dense_2/Tensordot/free:output:0.model/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
"model/dense_2/Tensordot/GatherV2_1GatherV2&model/dense_2/Tensordot/Shape:output:0%model/dense_2/Tensordot/axes:output:00model/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model/dense_2/Tensordot/ProdProd)model/dense_2/Tensordot/GatherV2:output:0&model/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
model/dense_2/Tensordot/Prod_1Prod+model/dense_2/Tensordot/GatherV2_1:output:0(model/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
model/dense_2/Tensordot/concatConcatV2%model/dense_2/Tensordot/free:output:0%model/dense_2/Tensordot/axes:output:0,model/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
model/dense_2/Tensordot/stackPack%model/dense_2/Tensordot/Prod:output:0'model/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
!model/dense_2/Tensordot/transpose	Transpose!model/dropout_2/Identity:output:0'model/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������-�
model/dense_2/Tensordot/ReshapeReshape%model/dense_2/Tensordot/transpose:y:0&model/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
model/dense_2/Tensordot/MatMulMatMul(model/dense_2/Tensordot/Reshape:output:0.model/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������j
model/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�g
%model/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model/dense_2/Tensordot/concat_1ConcatV2)model/dense_2/Tensordot/GatherV2:output:0(model/dense_2/Tensordot/Const_2:output:0.model/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
model/dense_2/TensordotReshape(model/dense_2/Tensordot/MatMul:product:0)model/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_2/BiasAddBiasAdd model/dense_2/Tensordot:output:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������q
model/dense_2/SeluSelumodel/dense_2/BiasAdd:output:0*
T0*,
_output_shapes
:����������}
model/dropout_3/IdentityIdentity model/dense_2/Selu:activations:0*
T0*,
_output_shapes
:�����������
&model/dense_3/Tensordot/ReadVariableOpReadVariableOp/model_dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0f
model/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
model/dense_3/Tensordot/ShapeShape!model/dropout_3/Identity:output:0*
T0*
_output_shapes
:g
%model/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model/dense_3/Tensordot/GatherV2GatherV2&model/dense_3/Tensordot/Shape:output:0%model/dense_3/Tensordot/free:output:0.model/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
"model/dense_3/Tensordot/GatherV2_1GatherV2&model/dense_3/Tensordot/Shape:output:0%model/dense_3/Tensordot/axes:output:00model/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model/dense_3/Tensordot/ProdProd)model/dense_3/Tensordot/GatherV2:output:0&model/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
model/dense_3/Tensordot/Prod_1Prod+model/dense_3/Tensordot/GatherV2_1:output:0(model/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
model/dense_3/Tensordot/concatConcatV2%model/dense_3/Tensordot/free:output:0%model/dense_3/Tensordot/axes:output:0,model/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
model/dense_3/Tensordot/stackPack%model/dense_3/Tensordot/Prod:output:0'model/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
!model/dense_3/Tensordot/transpose	Transpose!model/dropout_3/Identity:output:0'model/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
model/dense_3/Tensordot/ReshapeReshape%model/dense_3/Tensordot/transpose:y:0&model/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
model/dense_3/Tensordot/MatMulMatMul(model/dense_3/Tensordot/Reshape:output:0.model/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������j
model/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�g
%model/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model/dense_3/Tensordot/concat_1ConcatV2)model/dense_3/Tensordot/GatherV2:output:0(model/dense_3/Tensordot/Const_2:output:0.model/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
model/dense_3/TensordotReshape(model/dense_3/Tensordot/MatMul:product:0)model/dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_3/BiasAddBiasAdd model/dense_3/Tensordot:output:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������q
model/dense_3/SeluSelumodel/dense_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������}
model/dropout_4/IdentityIdentity model/dense_3/Selu:activations:0*
T0*,
_output_shapes
:�����������
&model/dense_4/Tensordot/ReadVariableOpReadVariableOp/model_dense_4_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype0f
model/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
model/dense_4/Tensordot/ShapeShape!model/dropout_4/Identity:output:0*
T0*
_output_shapes
:g
%model/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model/dense_4/Tensordot/GatherV2GatherV2&model/dense_4/Tensordot/Shape:output:0%model/dense_4/Tensordot/free:output:0.model/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
"model/dense_4/Tensordot/GatherV2_1GatherV2&model/dense_4/Tensordot/Shape:output:0%model/dense_4/Tensordot/axes:output:00model/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model/dense_4/Tensordot/ProdProd)model/dense_4/Tensordot/GatherV2:output:0&model/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
model/dense_4/Tensordot/Prod_1Prod+model/dense_4/Tensordot/GatherV2_1:output:0(model/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
model/dense_4/Tensordot/concatConcatV2%model/dense_4/Tensordot/free:output:0%model/dense_4/Tensordot/axes:output:0,model/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
model/dense_4/Tensordot/stackPack%model/dense_4/Tensordot/Prod:output:0'model/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
!model/dense_4/Tensordot/transpose	Transpose!model/dropout_4/Identity:output:0'model/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
model/dense_4/Tensordot/ReshapeReshape%model/dense_4/Tensordot/transpose:y:0&model/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
model/dense_4/Tensordot/MatMulMatMul(model/dense_4/Tensordot/Reshape:output:0.model/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@i
model/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@g
%model/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model/dense_4/Tensordot/concat_1ConcatV2)model/dense_4/Tensordot/GatherV2:output:0(model/dense_4/Tensordot/Const_2:output:0.model/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
model/dense_4/TensordotReshape(model/dense_4/Tensordot/MatMul:product:0)model/dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������@�
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense_4/BiasAddBiasAdd model/dense_4/Tensordot:output:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@p
model/dense_4/SeluSelumodel/dense_4/BiasAdd:output:0*
T0*+
_output_shapes
:���������@|
model/dropout_5/IdentityIdentity model/dense_4/Selu:activations:0*
T0*+
_output_shapes
:���������@�
&model/dense_5/Tensordot/ReadVariableOpReadVariableOp/model_dense_5_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0f
model/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
model/dense_5/Tensordot/ShapeShape!model/dropout_5/Identity:output:0*
T0*
_output_shapes
:g
%model/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model/dense_5/Tensordot/GatherV2GatherV2&model/dense_5/Tensordot/Shape:output:0%model/dense_5/Tensordot/free:output:0.model/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
"model/dense_5/Tensordot/GatherV2_1GatherV2&model/dense_5/Tensordot/Shape:output:0%model/dense_5/Tensordot/axes:output:00model/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model/dense_5/Tensordot/ProdProd)model/dense_5/Tensordot/GatherV2:output:0&model/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
model/dense_5/Tensordot/Prod_1Prod+model/dense_5/Tensordot/GatherV2_1:output:0(model/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
model/dense_5/Tensordot/concatConcatV2%model/dense_5/Tensordot/free:output:0%model/dense_5/Tensordot/axes:output:0,model/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
model/dense_5/Tensordot/stackPack%model/dense_5/Tensordot/Prod:output:0'model/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
!model/dense_5/Tensordot/transpose	Transpose!model/dropout_5/Identity:output:0'model/dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������@�
model/dense_5/Tensordot/ReshapeReshape%model/dense_5/Tensordot/transpose:y:0&model/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
model/dense_5/Tensordot/MatMulMatMul(model/dense_5/Tensordot/Reshape:output:0.model/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
model/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:g
%model/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model/dense_5/Tensordot/concat_1ConcatV2)model/dense_5/Tensordot/GatherV2:output:0(model/dense_5/Tensordot/Const_2:output:0.model/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
model/dense_5/TensordotReshape(model/dense_5/Tensordot/MatMul:product:0)model/dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:����������
$model/dense_5/BiasAdd/ReadVariableOpReadVariableOp-model_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_5/BiasAddBiasAdd model/dense_5/Tensordot:output:0,model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������p
model/dense_5/SeluSelumodel/dense_5/BiasAdd:output:0*
T0*+
_output_shapes
:���������|
model/dropout_6/IdentityIdentity model/dense_5/Selu:activations:0*
T0*+
_output_shapes
:����������
&model/dense_6/Tensordot/ReadVariableOpReadVariableOp/model_dense_6_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0f
model/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
model/dense_6/Tensordot/ShapeShape!model/dropout_6/Identity:output:0*
T0*
_output_shapes
:g
%model/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model/dense_6/Tensordot/GatherV2GatherV2&model/dense_6/Tensordot/Shape:output:0%model/dense_6/Tensordot/free:output:0.model/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
"model/dense_6/Tensordot/GatherV2_1GatherV2&model/dense_6/Tensordot/Shape:output:0%model/dense_6/Tensordot/axes:output:00model/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model/dense_6/Tensordot/ProdProd)model/dense_6/Tensordot/GatherV2:output:0&model/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
model/dense_6/Tensordot/Prod_1Prod+model/dense_6/Tensordot/GatherV2_1:output:0(model/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
model/dense_6/Tensordot/concatConcatV2%model/dense_6/Tensordot/free:output:0%model/dense_6/Tensordot/axes:output:0,model/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
model/dense_6/Tensordot/stackPack%model/dense_6/Tensordot/Prod:output:0'model/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
!model/dense_6/Tensordot/transpose	Transpose!model/dropout_6/Identity:output:0'model/dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:����������
model/dense_6/Tensordot/ReshapeReshape%model/dense_6/Tensordot/transpose:y:0&model/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
model/dense_6/Tensordot/MatMulMatMul(model/dense_6/Tensordot/Reshape:output:0.model/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
model/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:g
%model/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model/dense_6/Tensordot/concat_1ConcatV2)model/dense_6/Tensordot/GatherV2:output:0(model/dense_6/Tensordot/Const_2:output:0.model/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
model/dense_6/TensordotReshape(model/dense_6/Tensordot/MatMul:product:0)model/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:����������
$model/dense_6/BiasAdd/ReadVariableOpReadVariableOp-model_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_6/BiasAddBiasAdd model/dense_6/Tensordot:output:0,model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������p
model/dense_6/SeluSelumodel/dense_6/BiasAdd:output:0*
T0*+
_output_shapes
:���������s
IdentityIdentity model/dense_6/Selu:activations:0^NoOp*
T0*+
_output_shapes
:����������4
NoOpNoOp%^model/dense_2/BiasAdd/ReadVariableOp'^model/dense_2/Tensordot/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp'^model/dense_3/Tensordot/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp'^model/dense_4/Tensordot/ReadVariableOp%^model/dense_5/BiasAdd/ReadVariableOp'^model/dense_5/Tensordot/ReadVariableOp%^model/dense_6/BiasAdd/ReadVariableOp'^model/dense_6/Tensordot/ReadVariableOp>^model/token_and_position_embedding/embedding/embedding_lookup@^model/token_and_position_embedding/embedding_1/embedding_lookupE^model/transformer_block/layer_normalization/batchnorm/ReadVariableOpI^model/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpG^model/transformer_block/layer_normalization/batchnorm_1/ReadVariableOpK^model/transformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOpG^model/transformer_block/layer_normalization/batchnorm_2/ReadVariableOpK^model/transformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOpG^model/transformer_block/layer_normalization/batchnorm_3/ReadVariableOpK^model/transformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOpG^model/transformer_block/layer_normalization/batchnorm_4/ReadVariableOpK^model/transformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOpG^model/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpK^model/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpI^model/transformer_block/layer_normalization_1/batchnorm_1/ReadVariableOpM^model/transformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOpI^model/transformer_block/layer_normalization_1/batchnorm_2/ReadVariableOpM^model/transformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOpI^model/transformer_block/layer_normalization_1/batchnorm_3/ReadVariableOpM^model/transformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOpI^model/transformer_block/layer_normalization_1/batchnorm_4/ReadVariableOpM^model/transformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOpQ^model/transformer_block/multi_head_attention/attention_output/add/ReadVariableOpS^model/transformer_block/multi_head_attention/attention_output/add_1/ReadVariableOpS^model/transformer_block/multi_head_attention/attention_output/add_2/ReadVariableOpS^model/transformer_block/multi_head_attention/attention_output/add_3/ReadVariableOpS^model/transformer_block/multi_head_attention/attention_output/add_4/ReadVariableOp[^model/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp]^model/transformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOp]^model/transformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOp]^model/transformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOp]^model/transformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOpD^model/transformer_block/multi_head_attention/key/add/ReadVariableOpF^model/transformer_block/multi_head_attention/key/add_1/ReadVariableOpF^model/transformer_block/multi_head_attention/key/add_2/ReadVariableOpF^model/transformer_block/multi_head_attention/key/add_3/ReadVariableOpF^model/transformer_block/multi_head_attention/key/add_4/ReadVariableOpN^model/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpP^model/transformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOpP^model/transformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOpP^model/transformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOpP^model/transformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOpF^model/transformer_block/multi_head_attention/query/add/ReadVariableOpH^model/transformer_block/multi_head_attention/query/add_1/ReadVariableOpH^model/transformer_block/multi_head_attention/query/add_2/ReadVariableOpH^model/transformer_block/multi_head_attention/query/add_3/ReadVariableOpH^model/transformer_block/multi_head_attention/query/add_4/ReadVariableOpP^model/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpR^model/transformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOpR^model/transformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOpR^model/transformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOpR^model/transformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOpF^model/transformer_block/multi_head_attention/value/add/ReadVariableOpH^model/transformer_block/multi_head_attention/value/add_1/ReadVariableOpH^model/transformer_block/multi_head_attention/value/add_2/ReadVariableOpH^model/transformer_block/multi_head_attention/value/add_3/ReadVariableOpH^model/transformer_block/multi_head_attention/value/add_4/ReadVariableOpP^model/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpR^model/transformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOpR^model/transformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOpR^model/transformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOpR^model/transformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOp@^model/transformer_block/sequential/dense/BiasAdd/ReadVariableOpB^model/transformer_block/sequential/dense/BiasAdd_1/ReadVariableOpB^model/transformer_block/sequential/dense/BiasAdd_2/ReadVariableOpB^model/transformer_block/sequential/dense/BiasAdd_3/ReadVariableOpB^model/transformer_block/sequential/dense/BiasAdd_4/ReadVariableOpB^model/transformer_block/sequential/dense/Tensordot/ReadVariableOpD^model/transformer_block/sequential/dense/Tensordot_1/ReadVariableOpD^model/transformer_block/sequential/dense/Tensordot_2/ReadVariableOpD^model/transformer_block/sequential/dense/Tensordot_3/ReadVariableOpD^model/transformer_block/sequential/dense/Tensordot_4/ReadVariableOpB^model/transformer_block/sequential/dense_1/BiasAdd/ReadVariableOpD^model/transformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOpD^model/transformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOpD^model/transformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOpD^model/transformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOpD^model/transformer_block/sequential/dense_1/Tensordot/ReadVariableOpF^model/transformer_block/sequential/dense_1/Tensordot_1/ReadVariableOpF^model/transformer_block/sequential/dense_1/Tensordot_2/ReadVariableOpF^model/transformer_block/sequential/dense_1/Tensordot_3/ReadVariableOpF^model/transformer_block/sequential/dense_1/Tensordot_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������-: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2P
&model/dense_2/Tensordot/ReadVariableOp&model/dense_2/Tensordot/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2P
&model/dense_3/Tensordot/ReadVariableOp&model/dense_3/Tensordot/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2P
&model/dense_4/Tensordot/ReadVariableOp&model/dense_4/Tensordot/ReadVariableOp2L
$model/dense_5/BiasAdd/ReadVariableOp$model/dense_5/BiasAdd/ReadVariableOp2P
&model/dense_5/Tensordot/ReadVariableOp&model/dense_5/Tensordot/ReadVariableOp2L
$model/dense_6/BiasAdd/ReadVariableOp$model/dense_6/BiasAdd/ReadVariableOp2P
&model/dense_6/Tensordot/ReadVariableOp&model/dense_6/Tensordot/ReadVariableOp2~
=model/token_and_position_embedding/embedding/embedding_lookup=model/token_and_position_embedding/embedding/embedding_lookup2�
?model/token_and_position_embedding/embedding_1/embedding_lookup?model/token_and_position_embedding/embedding_1/embedding_lookup2�
Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOpDmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp2�
Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpHmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2�
Fmodel/transformer_block/layer_normalization/batchnorm_1/ReadVariableOpFmodel/transformer_block/layer_normalization/batchnorm_1/ReadVariableOp2�
Jmodel/transformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOpJmodel/transformer_block/layer_normalization/batchnorm_1/mul/ReadVariableOp2�
Fmodel/transformer_block/layer_normalization/batchnorm_2/ReadVariableOpFmodel/transformer_block/layer_normalization/batchnorm_2/ReadVariableOp2�
Jmodel/transformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOpJmodel/transformer_block/layer_normalization/batchnorm_2/mul/ReadVariableOp2�
Fmodel/transformer_block/layer_normalization/batchnorm_3/ReadVariableOpFmodel/transformer_block/layer_normalization/batchnorm_3/ReadVariableOp2�
Jmodel/transformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOpJmodel/transformer_block/layer_normalization/batchnorm_3/mul/ReadVariableOp2�
Fmodel/transformer_block/layer_normalization/batchnorm_4/ReadVariableOpFmodel/transformer_block/layer_normalization/batchnorm_4/ReadVariableOp2�
Jmodel/transformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOpJmodel/transformer_block/layer_normalization/batchnorm_4/mul/ReadVariableOp2�
Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpFmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2�
Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpJmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2�
Hmodel/transformer_block/layer_normalization_1/batchnorm_1/ReadVariableOpHmodel/transformer_block/layer_normalization_1/batchnorm_1/ReadVariableOp2�
Lmodel/transformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOpLmodel/transformer_block/layer_normalization_1/batchnorm_1/mul/ReadVariableOp2�
Hmodel/transformer_block/layer_normalization_1/batchnorm_2/ReadVariableOpHmodel/transformer_block/layer_normalization_1/batchnorm_2/ReadVariableOp2�
Lmodel/transformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOpLmodel/transformer_block/layer_normalization_1/batchnorm_2/mul/ReadVariableOp2�
Hmodel/transformer_block/layer_normalization_1/batchnorm_3/ReadVariableOpHmodel/transformer_block/layer_normalization_1/batchnorm_3/ReadVariableOp2�
Lmodel/transformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOpLmodel/transformer_block/layer_normalization_1/batchnorm_3/mul/ReadVariableOp2�
Hmodel/transformer_block/layer_normalization_1/batchnorm_4/ReadVariableOpHmodel/transformer_block/layer_normalization_1/batchnorm_4/ReadVariableOp2�
Lmodel/transformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOpLmodel/transformer_block/layer_normalization_1/batchnorm_4/mul/ReadVariableOp2�
Pmodel/transformer_block/multi_head_attention/attention_output/add/ReadVariableOpPmodel/transformer_block/multi_head_attention/attention_output/add/ReadVariableOp2�
Rmodel/transformer_block/multi_head_attention/attention_output/add_1/ReadVariableOpRmodel/transformer_block/multi_head_attention/attention_output/add_1/ReadVariableOp2�
Rmodel/transformer_block/multi_head_attention/attention_output/add_2/ReadVariableOpRmodel/transformer_block/multi_head_attention/attention_output/add_2/ReadVariableOp2�
Rmodel/transformer_block/multi_head_attention/attention_output/add_3/ReadVariableOpRmodel/transformer_block/multi_head_attention/attention_output/add_3/ReadVariableOp2�
Rmodel/transformer_block/multi_head_attention/attention_output/add_4/ReadVariableOpRmodel/transformer_block/multi_head_attention/attention_output/add_4/ReadVariableOp2�
Zmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpZmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2�
\model/transformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOp\model/transformer_block/multi_head_attention/attention_output/einsum_1/Einsum/ReadVariableOp2�
\model/transformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOp\model/transformer_block/multi_head_attention/attention_output/einsum_2/Einsum/ReadVariableOp2�
\model/transformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOp\model/transformer_block/multi_head_attention/attention_output/einsum_3/Einsum/ReadVariableOp2�
\model/transformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOp\model/transformer_block/multi_head_attention/attention_output/einsum_4/Einsum/ReadVariableOp2�
Cmodel/transformer_block/multi_head_attention/key/add/ReadVariableOpCmodel/transformer_block/multi_head_attention/key/add/ReadVariableOp2�
Emodel/transformer_block/multi_head_attention/key/add_1/ReadVariableOpEmodel/transformer_block/multi_head_attention/key/add_1/ReadVariableOp2�
Emodel/transformer_block/multi_head_attention/key/add_2/ReadVariableOpEmodel/transformer_block/multi_head_attention/key/add_2/ReadVariableOp2�
Emodel/transformer_block/multi_head_attention/key/add_3/ReadVariableOpEmodel/transformer_block/multi_head_attention/key/add_3/ReadVariableOp2�
Emodel/transformer_block/multi_head_attention/key/add_4/ReadVariableOpEmodel/transformer_block/multi_head_attention/key/add_4/ReadVariableOp2�
Mmodel/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpMmodel/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp2�
Omodel/transformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOpOmodel/transformer_block/multi_head_attention/key/einsum_1/Einsum/ReadVariableOp2�
Omodel/transformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOpOmodel/transformer_block/multi_head_attention/key/einsum_2/Einsum/ReadVariableOp2�
Omodel/transformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOpOmodel/transformer_block/multi_head_attention/key/einsum_3/Einsum/ReadVariableOp2�
Omodel/transformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOpOmodel/transformer_block/multi_head_attention/key/einsum_4/Einsum/ReadVariableOp2�
Emodel/transformer_block/multi_head_attention/query/add/ReadVariableOpEmodel/transformer_block/multi_head_attention/query/add/ReadVariableOp2�
Gmodel/transformer_block/multi_head_attention/query/add_1/ReadVariableOpGmodel/transformer_block/multi_head_attention/query/add_1/ReadVariableOp2�
Gmodel/transformer_block/multi_head_attention/query/add_2/ReadVariableOpGmodel/transformer_block/multi_head_attention/query/add_2/ReadVariableOp2�
Gmodel/transformer_block/multi_head_attention/query/add_3/ReadVariableOpGmodel/transformer_block/multi_head_attention/query/add_3/ReadVariableOp2�
Gmodel/transformer_block/multi_head_attention/query/add_4/ReadVariableOpGmodel/transformer_block/multi_head_attention/query/add_4/ReadVariableOp2�
Omodel/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpOmodel/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp2�
Qmodel/transformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOpQmodel/transformer_block/multi_head_attention/query/einsum_1/Einsum/ReadVariableOp2�
Qmodel/transformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOpQmodel/transformer_block/multi_head_attention/query/einsum_2/Einsum/ReadVariableOp2�
Qmodel/transformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOpQmodel/transformer_block/multi_head_attention/query/einsum_3/Einsum/ReadVariableOp2�
Qmodel/transformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOpQmodel/transformer_block/multi_head_attention/query/einsum_4/Einsum/ReadVariableOp2�
Emodel/transformer_block/multi_head_attention/value/add/ReadVariableOpEmodel/transformer_block/multi_head_attention/value/add/ReadVariableOp2�
Gmodel/transformer_block/multi_head_attention/value/add_1/ReadVariableOpGmodel/transformer_block/multi_head_attention/value/add_1/ReadVariableOp2�
Gmodel/transformer_block/multi_head_attention/value/add_2/ReadVariableOpGmodel/transformer_block/multi_head_attention/value/add_2/ReadVariableOp2�
Gmodel/transformer_block/multi_head_attention/value/add_3/ReadVariableOpGmodel/transformer_block/multi_head_attention/value/add_3/ReadVariableOp2�
Gmodel/transformer_block/multi_head_attention/value/add_4/ReadVariableOpGmodel/transformer_block/multi_head_attention/value/add_4/ReadVariableOp2�
Omodel/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpOmodel/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp2�
Qmodel/transformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOpQmodel/transformer_block/multi_head_attention/value/einsum_1/Einsum/ReadVariableOp2�
Qmodel/transformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOpQmodel/transformer_block/multi_head_attention/value/einsum_2/Einsum/ReadVariableOp2�
Qmodel/transformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOpQmodel/transformer_block/multi_head_attention/value/einsum_3/Einsum/ReadVariableOp2�
Qmodel/transformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOpQmodel/transformer_block/multi_head_attention/value/einsum_4/Einsum/ReadVariableOp2�
?model/transformer_block/sequential/dense/BiasAdd/ReadVariableOp?model/transformer_block/sequential/dense/BiasAdd/ReadVariableOp2�
Amodel/transformer_block/sequential/dense/BiasAdd_1/ReadVariableOpAmodel/transformer_block/sequential/dense/BiasAdd_1/ReadVariableOp2�
Amodel/transformer_block/sequential/dense/BiasAdd_2/ReadVariableOpAmodel/transformer_block/sequential/dense/BiasAdd_2/ReadVariableOp2�
Amodel/transformer_block/sequential/dense/BiasAdd_3/ReadVariableOpAmodel/transformer_block/sequential/dense/BiasAdd_3/ReadVariableOp2�
Amodel/transformer_block/sequential/dense/BiasAdd_4/ReadVariableOpAmodel/transformer_block/sequential/dense/BiasAdd_4/ReadVariableOp2�
Amodel/transformer_block/sequential/dense/Tensordot/ReadVariableOpAmodel/transformer_block/sequential/dense/Tensordot/ReadVariableOp2�
Cmodel/transformer_block/sequential/dense/Tensordot_1/ReadVariableOpCmodel/transformer_block/sequential/dense/Tensordot_1/ReadVariableOp2�
Cmodel/transformer_block/sequential/dense/Tensordot_2/ReadVariableOpCmodel/transformer_block/sequential/dense/Tensordot_2/ReadVariableOp2�
Cmodel/transformer_block/sequential/dense/Tensordot_3/ReadVariableOpCmodel/transformer_block/sequential/dense/Tensordot_3/ReadVariableOp2�
Cmodel/transformer_block/sequential/dense/Tensordot_4/ReadVariableOpCmodel/transformer_block/sequential/dense/Tensordot_4/ReadVariableOp2�
Amodel/transformer_block/sequential/dense_1/BiasAdd/ReadVariableOpAmodel/transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp2�
Cmodel/transformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOpCmodel/transformer_block/sequential/dense_1/BiasAdd_1/ReadVariableOp2�
Cmodel/transformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOpCmodel/transformer_block/sequential/dense_1/BiasAdd_2/ReadVariableOp2�
Cmodel/transformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOpCmodel/transformer_block/sequential/dense_1/BiasAdd_3/ReadVariableOp2�
Cmodel/transformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOpCmodel/transformer_block/sequential/dense_1/BiasAdd_4/ReadVariableOp2�
Cmodel/transformer_block/sequential/dense_1/Tensordot/ReadVariableOpCmodel/transformer_block/sequential/dense_1/Tensordot/ReadVariableOp2�
Emodel/transformer_block/sequential/dense_1/Tensordot_1/ReadVariableOpEmodel/transformer_block/sequential/dense_1/Tensordot_1/ReadVariableOp2�
Emodel/transformer_block/sequential/dense_1/Tensordot_2/ReadVariableOpEmodel/transformer_block/sequential/dense_1/Tensordot_2/ReadVariableOp2�
Emodel/transformer_block/sequential/dense_1/Tensordot_3/ReadVariableOpEmodel/transformer_block/sequential/dense_1/Tensordot_3/ReadVariableOp2�
Emodel/transformer_block/sequential/dense_1/Tensordot_4/ReadVariableOpEmodel/transformer_block/sequential/dense_1/Tensordot_4/ReadVariableOp:P L
'
_output_shapes
:���������-
!
_user_specified_name	input_1
�

b
C__inference_dropout_2_layer_call_and_return_conditional_losses_5799

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������-C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������-*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * gL=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������-t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������-n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������-^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������-:T P
,
_output_shapes
:����������-
 
_user_specified_nameinputs
�
a
(__inference_dropout_4_layer_call_fn_5916

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_2741t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�A
�
__inference__traced_save_6433
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableopP
Lsavev2_token_and_position_embedding_embedding_embeddings_read_readvariableopR
Nsavev2_token_and_position_embedding_embedding_1_embeddings_read_readvariableopR
Nsavev2_transformer_block_multi_head_attention_query_kernel_read_readvariableopP
Lsavev2_transformer_block_multi_head_attention_query_bias_read_readvariableopP
Lsavev2_transformer_block_multi_head_attention_key_kernel_read_readvariableopN
Jsavev2_transformer_block_multi_head_attention_key_bias_read_readvariableopR
Nsavev2_transformer_block_multi_head_attention_value_kernel_read_readvariableopP
Lsavev2_transformer_block_multi_head_attention_value_bias_read_readvariableop]
Ysavev2_transformer_block_multi_head_attention_attention_output_kernel_read_readvariableop[
Wsavev2_transformer_block_multi_head_attention_attention_output_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableopJ
Fsavev2_transformer_block_layer_normalization_gamma_read_readvariableopI
Esavev2_transformer_block_layer_normalization_beta_read_readvariableopL
Hsavev2_transformer_block_layer_normalization_1_gamma_read_readvariableopK
Gsavev2_transformer_block_layer_normalization_1_beta_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableopLsavev2_token_and_position_embedding_embedding_embeddings_read_readvariableopNsavev2_token_and_position_embedding_embedding_1_embeddings_read_readvariableopNsavev2_transformer_block_multi_head_attention_query_kernel_read_readvariableopLsavev2_transformer_block_multi_head_attention_query_bias_read_readvariableopLsavev2_transformer_block_multi_head_attention_key_kernel_read_readvariableopJsavev2_transformer_block_multi_head_attention_key_bias_read_readvariableopNsavev2_transformer_block_multi_head_attention_value_kernel_read_readvariableopLsavev2_transformer_block_multi_head_attention_value_bias_read_readvariableopYsavev2_transformer_block_multi_head_attention_attention_output_kernel_read_readvariableopWsavev2_transformer_block_multi_head_attention_attention_output_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopFsavev2_transformer_block_layer_normalization_gamma_read_readvariableopEsavev2_transformer_block_layer_normalization_beta_read_readvariableopHsavev2_transformer_block_layer_normalization_1_gamma_read_readvariableopGsavev2_transformer_block_layer_normalization_1_beta_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
�-�:�:
��:�:	�@:@:@::::
��:	-�:��:	�:��:	�:��:	�:��:�:
��:�:
��:�:�:�:�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
�-�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::&"
 
_output_shapes
:
��:%!

_output_shapes
:	-�:*&
$
_output_shapes
:��:%!

_output_shapes
:	�:*&
$
_output_shapes
:��:%!

_output_shapes
:	�:*&
$
_output_shapes
:��:%!

_output_shapes
:	�:*&
$
_output_shapes
:��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:

_output_shapes
: 
�
�
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_5413
x4
!embedding_1_embedding_lookup_5400:	-�3
embedding_embedding_lookup_5406:
��
identity��embedding/embedding_lookup�embedding_1/embedding_lookup6
ShapeShapex*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*
_output_shapes
:-�
embedding_1/embedding_lookupResourceGather!embedding_1_embedding_lookup_5400range:output:0*
Tindices0*4
_class*
(&loc:@embedding_1/embedding_lookup/5400*
_output_shapes
:	-�*
dtype0�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding_1/embedding_lookup/5400*
_output_shapes
:	-��
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	-�Z
embedding/CastCastx*

DstT0*

SrcT0*'
_output_shapes
:���������-�
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_5406embedding/Cast:y:0*
Tindices0*2
_class(
&$loc:@embedding/embedding_lookup/5406*,
_output_shapes
:���������-�*
dtype0�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/5406*,
_output_shapes
:���������-��
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:���������-��
addAddV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:���������-�[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:���������-��
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:J F
'
_output_shapes
:���������-

_user_specified_namex
�
a
C__inference_dropout_3_layer_call_and_return_conditional_losses_2414

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_2645
input_1
unknown:	-�
	unknown_0:
��!
	unknown_1:��
	unknown_2:	�!
	unknown_3:��
	unknown_4:	�!
	unknown_5:��
	unknown_6:	�!
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
�-�

unknown_18:	�

unknown_19:
��

unknown_20:	�

unknown_21:	�@

unknown_22:@

unknown_23:@

unknown_24:

unknown_25:

unknown_26:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*>
_read_only_resource_inputs 
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_2586s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������-: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������-
!
_user_specified_name	input_1
��
�
K__inference_transformer_block_layer_call_and_return_conditional_losses_2996

inputsX
@multi_head_attention_query_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_query_add_readvariableop_resource:	�V
>multi_head_attention_key_einsum_einsum_readvariableop_resource:��G
4multi_head_attention_key_add_readvariableop_resource:	�X
@multi_head_attention_value_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_value_add_readvariableop_resource:	�c
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:��P
Amulti_head_attention_attention_output_add_readvariableop_resource:	�H
9layer_normalization_batchnorm_mul_readvariableop_resource:	�D
5layer_normalization_batchnorm_readvariableop_resource:	�F
2sequential_dense_tensordot_readvariableop_resource:
��?
0sequential_dense_biasadd_readvariableop_resource:	�H
4sequential_dense_1_tensordot_readvariableop_resource:
��A
2sequential_dense_1_biasadd_readvariableop_resource:	�J
;layer_normalization_1_batchnorm_mul_readvariableop_resource:	�F
7layer_normalization_1_batchnorm_readvariableop_resource:	�
identity��,layer_normalization/batchnorm/ReadVariableOp�0layer_normalization/batchnorm/mul/ReadVariableOp�.layer_normalization_1/batchnorm/ReadVariableOp�2layer_normalization_1/batchnorm/mul/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOp�'sequential/dense/BiasAdd/ReadVariableOp�)sequential/dense/Tensordot/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�+sequential/dense_1/Tensordot/ReadVariableOp�
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:���������-��
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������--�
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/softmax/Softmax:softmax:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout/dropout/MulMul-multi_head_attention/attention_output/add:z:0dropout/dropout/Const:output:0*
T0*,
_output_shapes
:���������-�r
dropout/dropout/ShapeShape-multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:���������-�*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *U;>�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������-��
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������-��
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:���������-�f
addAddV2inputsdropout/dropout/Mul_1:z:0*
T0*,
_output_shapes
:���������-�|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:���������-�
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������-�
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:���������-�
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������-��
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������-��
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       w
 sequential/dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$sequential/dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0*sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-��
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�w
sequential/dense/SeluSelu!sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:���������-��
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0k
!sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       u
"sequential/dense_1/Tensordot/ShapeShape#sequential/dense/Selu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense_1/Tensordot/GatherV2GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/free:output:03sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential/dense_1/Tensordot/GatherV2_1GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/axes:output:05sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
&sequential/dense_1/Tensordot/transpose	Transpose#sequential/dense/Selu:activations:0,sequential/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������o
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-��
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout_1/dropout/MulMul#sequential/dense_1/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:���������-�j
dropout_1/dropout/ShapeShape#sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:�
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:���������-�*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *U;>�
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������-��
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������-��
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:���������-��
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*,
_output_shapes
:���������-�~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:���������-�
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������-�
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:���������-�
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������-��
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������-�}
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������-��
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������-�: : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�
a
(__inference_dropout_2_layer_call_fn_5782

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������-* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_2807t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������-22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������-
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_3771

inputs
unknown:	-�
	unknown_0:
��!
	unknown_1:��
	unknown_2:	�!
	unknown_3:��
	unknown_4:	�!
	unknown_5:��
	unknown_6:	�!
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
�-�

unknown_18:	�

unknown_19:
��

unknown_20:	�

unknown_21:	�@

unknown_22:@

unknown_23:@

unknown_24:

unknown_25:

unknown_26:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*>
_read_only_resource_inputs 
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_2586s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������-: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�
a
C__inference_dropout_6_layer_call_and_return_conditional_losses_2546

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

b
C__inference_dropout_6_layer_call_and_return_conditional_losses_2675

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * gL=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
A__inference_dense_1_layer_call_and_return_conditional_losses_6326

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-�s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������-�z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������-�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�
�
A__inference_dense_5_layer_call_and_return_conditional_losses_2535

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������@�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
SeluSeluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentitySelu:activations:0^NoOp*
T0*+
_output_shapes
:���������z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�=
�
D__inference_sequential_layer_call_and_return_conditional_losses_6247

inputs;
'dense_tensordot_readvariableop_resource:
��4
%dense_biasadd_readvariableop_resource:	�=
)dense_1_tensordot_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       K
dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�a

dense/SeluSeludense/BiasAdd:output:0*
T0*,
_output_shapes
:���������-��
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapedense/Selu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_1/Tensordot/transpose	Transposedense/Selu:activations:0!dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-��
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�l
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������-��
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������-�: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
��
�
K__inference_transformer_block_layer_call_and_return_conditional_losses_5754

inputsX
@multi_head_attention_query_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_query_add_readvariableop_resource:	�V
>multi_head_attention_key_einsum_einsum_readvariableop_resource:��G
4multi_head_attention_key_add_readvariableop_resource:	�X
@multi_head_attention_value_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_value_add_readvariableop_resource:	�c
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:��P
Amulti_head_attention_attention_output_add_readvariableop_resource:	�H
9layer_normalization_batchnorm_mul_readvariableop_resource:	�D
5layer_normalization_batchnorm_readvariableop_resource:	�F
2sequential_dense_tensordot_readvariableop_resource:
��?
0sequential_dense_biasadd_readvariableop_resource:	�H
4sequential_dense_1_tensordot_readvariableop_resource:
��A
2sequential_dense_1_biasadd_readvariableop_resource:	�J
;layer_normalization_1_batchnorm_mul_readvariableop_resource:	�F
7layer_normalization_1_batchnorm_readvariableop_resource:	�
identity��,layer_normalization/batchnorm/ReadVariableOp�0layer_normalization/batchnorm/mul/ReadVariableOp�.layer_normalization_1/batchnorm/ReadVariableOp�2layer_normalization_1/batchnorm/mul/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOp�'sequential/dense/BiasAdd/ReadVariableOp�)sequential/dense/Tensordot/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�+sequential/dense_1/Tensordot/ReadVariableOp�
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-��
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������-�*
equationabc,cde->abde�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������-�_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:���������-��
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������--*
equationaecd,abcd->acbe�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������--�
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/softmax/Softmax:softmax:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:���������-�*
equationacbe,aecd->abcd�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������-�*
equationabcd,cde->abe�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout/dropout/MulMul-multi_head_attention/attention_output/add:z:0dropout/dropout/Const:output:0*
T0*,
_output_shapes
:���������-�r
dropout/dropout/ShapeShape-multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:���������-�*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *U;>�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������-��
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������-��
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:���������-�f
addAddV2inputsdropout/dropout/Mul_1:z:0*
T0*,
_output_shapes
:���������-�|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:���������-�
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������-�
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:���������-�
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������-��
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������-��
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       w
 sequential/dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$sequential/dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0*sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-��
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�w
sequential/dense/SeluSelu!sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:���������-��
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0k
!sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       u
"sequential/dense_1/Tensordot/ShapeShape#sequential/dense/Selu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense_1/Tensordot/GatherV2GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/free:output:03sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential/dense_1/Tensordot/GatherV2_1GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/axes:output:05sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
&sequential/dense_1/Tensordot/transpose	Transpose#sequential/dense/Selu:activations:0,sequential/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������o
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-��
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout_1/dropout/MulMul#sequential/dense_1/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:���������-�j
dropout_1/dropout/ShapeShape#sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:�
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:���������-�*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *U;>�
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������-��
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������-��
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:���������-��
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*,
_output_shapes
:���������-�~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(�
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:���������-�
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������-��
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������-*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������-�
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:���������-�
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-��
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������-��
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������-��
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������-�}
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������-��
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������-�: : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�
�
A__inference_dense_4_layer_call_and_return_conditional_losses_2491

inputs4
!tensordot_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@T
SeluSeluBiasAdd:output:0*
T0*+
_output_shapes
:���������@e
IdentityIdentitySelu:activations:0^NoOp*
T0*+
_output_shapes
:���������@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
0__inference_transformer_block_layer_call_fn_5487

inputs
unknown:��
	unknown_0:	�!
	unknown_1:��
	unknown_2:	�!
	unknown_3:��
	unknown_4:	�!
	unknown_5:��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*2
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_2996t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������-�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������-�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�
�
?__inference_dense_layer_call_and_return_conditional_losses_1929

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-�s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�U
SeluSeluBiasAdd:output:0*
T0*,
_output_shapes
:���������-�f
IdentityIdentitySelu:activations:0^NoOp*
T0*,
_output_shapes
:���������-�z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������-�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�
�
)__inference_sequential_layer_call_fn_6120

inputs
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1972t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������-�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������-�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�
a
C__inference_dropout_5_layer_call_and_return_conditional_losses_2502

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������@_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
A__inference_dense_3_layer_call_and_return_conditional_losses_2447

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
SeluSeluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentitySelu:activations:0^NoOp*
T0*,
_output_shapes
:����������z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_sequential_layer_call_fn_1983
dense_input
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������-�*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1972t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������-�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������-�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:���������-�
%
_user_specified_namedense_input
�s
�
 __inference__traced_restore_6527
file_prefix3
assignvariableop_dense_2_kernel:
�-�.
assignvariableop_1_dense_2_bias:	�5
!assignvariableop_2_dense_3_kernel:
��.
assignvariableop_3_dense_3_bias:	�4
!assignvariableop_4_dense_4_kernel:	�@-
assignvariableop_5_dense_4_bias:@3
!assignvariableop_6_dense_5_kernel:@-
assignvariableop_7_dense_5_bias:3
!assignvariableop_8_dense_6_kernel:-
assignvariableop_9_dense_6_bias:Y
Eassignvariableop_10_token_and_position_embedding_embedding_embeddings:
��Z
Gassignvariableop_11_token_and_position_embedding_embedding_1_embeddings:	-�_
Gassignvariableop_12_transformer_block_multi_head_attention_query_kernel:��X
Eassignvariableop_13_transformer_block_multi_head_attention_query_bias:	�]
Eassignvariableop_14_transformer_block_multi_head_attention_key_kernel:��V
Cassignvariableop_15_transformer_block_multi_head_attention_key_bias:	�_
Gassignvariableop_16_transformer_block_multi_head_attention_value_kernel:��X
Eassignvariableop_17_transformer_block_multi_head_attention_value_bias:	�j
Rassignvariableop_18_transformer_block_multi_head_attention_attention_output_kernel:��_
Passignvariableop_19_transformer_block_multi_head_attention_attention_output_bias:	�4
 assignvariableop_20_dense_kernel:
��-
assignvariableop_21_dense_bias:	�6
"assignvariableop_22_dense_1_kernel:
��/
 assignvariableop_23_dense_1_bias:	�N
?assignvariableop_24_transformer_block_layer_normalization_gamma:	�M
>assignvariableop_25_transformer_block_layer_normalization_beta:	�P
Aassignvariableop_26_transformer_block_layer_normalization_1_gamma:	�O
@assignvariableop_27_transformer_block_layer_normalization_1_beta:	�
identity_29��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_6_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_6_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpEassignvariableop_10_token_and_position_embedding_embedding_embeddingsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpGassignvariableop_11_token_and_position_embedding_embedding_1_embeddingsIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpGassignvariableop_12_transformer_block_multi_head_attention_query_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpEassignvariableop_13_transformer_block_multi_head_attention_query_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpEassignvariableop_14_transformer_block_multi_head_attention_key_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpCassignvariableop_15_transformer_block_multi_head_attention_key_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpGassignvariableop_16_transformer_block_multi_head_attention_value_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpEassignvariableop_17_transformer_block_multi_head_attention_value_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpRassignvariableop_18_transformer_block_multi_head_attention_attention_output_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpPassignvariableop_19_transformer_block_multi_head_attention_attention_output_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp assignvariableop_20_dense_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_dense_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_1_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_1_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp?assignvariableop_24_transformer_block_layer_normalization_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp>assignvariableop_25_transformer_block_layer_normalization_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpAassignvariableop_26_transformer_block_layer_normalization_1_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp@assignvariableop_27_transformer_block_layer_normalization_1_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
A__inference_dense_1_layer_call_and_return_conditional_losses_1965

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:���������-��
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������-�s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������-�d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������-�z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������-�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:���������-�
 
_user_specified_nameinputs
�

b
C__inference_dropout_5_layer_call_and_return_conditional_losses_6000

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * gL=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������@s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������@m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������@]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
A__inference_dense_2_layer_call_and_return_conditional_losses_2403

inputs5
!tensordot_readvariableop_resource:
�-�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
�-�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:����������-�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
SeluSeluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentitySelu:activations:0^NoOp*
T0*,
_output_shapes
:����������z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������-
 
_user_specified_nameinputs
�
a
C__inference_dropout_5_layer_call_and_return_conditional_losses_5988

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������@_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������-?
dense_64
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	token_emb
pos_emb"
_tf_keras_layer
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%att
&ffn
'
layernorm1
(
layernorm2
)dropout1
*dropout2"
_tf_keras_layer
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7_random_generator"
_tf_keras_layer
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias"
_tf_keras_layer
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
F_random_generator"
_tf_keras_layer
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

Mkernel
Nbias"
_tf_keras_layer
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
U_random_generator"
_tf_keras_layer
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias"
_tf_keras_layer
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
d_random_generator"
_tf_keras_layer
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias"
_tf_keras_layer
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
s_random_generator"
_tf_keras_layer
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

zkernel
{bias"
_tf_keras_layer
�
|0
}1
~2
3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
>18
?19
M20
N21
\22
]23
k24
l25
z26
{27"
trackable_list_wrapper
�
|0
}1
~2
3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
>18
?19
M20
N21
\22
]23
k24
l25
z26
{27"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
$__inference_model_layer_call_fn_2645
$__inference_model_layer_call_fn_3771
$__inference_model_layer_call_fn_3832
$__inference_model_layer_call_fn_3365�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
?__inference_model_layer_call_and_return_conditional_losses_4556
?__inference_model_layer_call_and_return_conditional_losses_5380
?__inference_model_layer_call_and_return_conditional_losses_3506
?__inference_model_layer_call_and_return_conditional_losses_3647�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
__inference__wrapped_model_1891input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
;__inference_token_and_position_embedding_layer_call_fn_5389�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_5413�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
|
embeddings"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
}
embeddings"
_tf_keras_layer
�
~0
1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�
~0
1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
0__inference_transformer_block_layer_call_fn_5450
0__inference_transformer_block_layer_call_fn_5487�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
K__inference_transformer_block_layer_call_and_return_conditional_losses_5614
K__inference_transformer_block_layer_call_and_return_conditional_losses_5754�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense"
_tf_keras_layer
�
�layer_with_weights-0
�layer-0
�layer_with_weights-1
�layer-1
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_reshape_layer_call_fn_5759�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_reshape_layer_call_and_return_conditional_losses_5772�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dropout_2_layer_call_fn_5777
(__inference_dropout_2_layer_call_fn_5782�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dropout_2_layer_call_and_return_conditional_losses_5787
C__inference_dropout_2_layer_call_and_return_conditional_losses_5799�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_2_layer_call_fn_5808�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_2_layer_call_and_return_conditional_losses_5839�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 
�-�2dense_2/kernel
:�2dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dropout_3_layer_call_fn_5844
(__inference_dropout_3_layer_call_fn_5849�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dropout_3_layer_call_and_return_conditional_losses_5854
C__inference_dropout_3_layer_call_and_return_conditional_losses_5866�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_3_layer_call_fn_5875�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_3_layer_call_and_return_conditional_losses_5906�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 
��2dense_3/kernel
:�2dense_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dropout_4_layer_call_fn_5911
(__inference_dropout_4_layer_call_fn_5916�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dropout_4_layer_call_and_return_conditional_losses_5921
C__inference_dropout_4_layer_call_and_return_conditional_losses_5933�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_4_layer_call_fn_5942�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_4_layer_call_and_return_conditional_losses_5973�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:	�@2dense_4/kernel
:@2dense_4/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dropout_5_layer_call_fn_5978
(__inference_dropout_5_layer_call_fn_5983�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dropout_5_layer_call_and_return_conditional_losses_5988
C__inference_dropout_5_layer_call_and_return_conditional_losses_6000�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_5_layer_call_fn_6009�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_5_layer_call_and_return_conditional_losses_6040�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :@2dense_5/kernel
:2dense_5/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dropout_6_layer_call_fn_6045
(__inference_dropout_6_layer_call_fn_6050�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dropout_6_layer_call_and_return_conditional_losses_6055
C__inference_dropout_6_layer_call_and_return_conditional_losses_6067�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_6_layer_call_fn_6076�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_6_layer_call_and_return_conditional_losses_6107�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :2dense_6/kernel
:2dense_6/bias
E:C
��21token_and_position_embedding/embedding/embeddings
F:D	-�23token_and_position_embedding/embedding_1/embeddings
K:I��23transformer_block/multi_head_attention/query/kernel
D:B	�21transformer_block/multi_head_attention/query/bias
I:G��21transformer_block/multi_head_attention/key/kernel
B:@	�2/transformer_block/multi_head_attention/key/bias
K:I��23transformer_block/multi_head_attention/value/kernel
D:B	�21transformer_block/multi_head_attention/value/bias
V:T��2>transformer_block/multi_head_attention/attention_output/kernel
K:I�2<transformer_block/multi_head_attention/attention_output/bias
 :
��2dense/kernel
:�2
dense/bias
": 
��2dense_1/kernel
:�2dense_1/bias
::8�2+transformer_block/layer_normalization/gamma
9:7�2*transformer_block/layer_normalization/beta
<::�2-transformer_block/layer_normalization_1/gamma
;:9�2,transformer_block/layer_normalization_1/beta
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_model_layer_call_fn_2645input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_3771inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_3832inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_3365input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_4556inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_5380inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_3506input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_3647input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_signature_wrapper_3710input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
;__inference_token_and_position_embedding_layer_call_fn_5389x"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_5413x"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
|0"
trackable_list_wrapper
'
|0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
}0"
trackable_list_wrapper
'
}0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
J
%0
&1
'2
(3
)4
*5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_transformer_block_layer_call_fn_5450inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
0__inference_transformer_block_layer_call_fn_5487inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_transformer_block_layer_call_and_return_conditional_losses_5614inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_transformer_block_layer_call_and_return_conditional_losses_5754inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
^
~0
1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
^
~0
1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpecx
argsp�m
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecx
argsp�m
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

~kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
)__inference_sequential_layer_call_fn_1983
)__inference_sequential_layer_call_fn_6120
)__inference_sequential_layer_call_fn_6133
)__inference_sequential_layer_call_fn_2056�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
D__inference_sequential_layer_call_and_return_conditional_losses_6190
D__inference_sequential_layer_call_and_return_conditional_losses_6247
D__inference_sequential_layer_call_and_return_conditional_losses_2070
D__inference_sequential_layer_call_and_return_conditional_losses_2084�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
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
�B�
&__inference_reshape_layer_call_fn_5759inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_reshape_layer_call_and_return_conditional_losses_5772inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_dropout_2_layer_call_fn_5777inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_dropout_2_layer_call_fn_5782inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_2_layer_call_and_return_conditional_losses_5787inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_2_layer_call_and_return_conditional_losses_5799inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
&__inference_dense_2_layer_call_fn_5808inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_2_layer_call_and_return_conditional_losses_5839inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_dropout_3_layer_call_fn_5844inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_dropout_3_layer_call_fn_5849inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_3_layer_call_and_return_conditional_losses_5854inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_3_layer_call_and_return_conditional_losses_5866inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
&__inference_dense_3_layer_call_fn_5875inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_3_layer_call_and_return_conditional_losses_5906inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_dropout_4_layer_call_fn_5911inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_dropout_4_layer_call_fn_5916inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_4_layer_call_and_return_conditional_losses_5921inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_4_layer_call_and_return_conditional_losses_5933inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
&__inference_dense_4_layer_call_fn_5942inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_4_layer_call_and_return_conditional_losses_5973inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_dropout_5_layer_call_fn_5978inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_dropout_5_layer_call_fn_5983inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_5_layer_call_and_return_conditional_losses_5988inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_5_layer_call_and_return_conditional_losses_6000inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
&__inference_dense_5_layer_call_fn_6009inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_5_layer_call_and_return_conditional_losses_6040inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_dropout_6_layer_call_fn_6045inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_dropout_6_layer_call_fn_6050inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_6_layer_call_and_return_conditional_losses_6055inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_6_layer_call_and_return_conditional_losses_6067inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
&__inference_dense_6_layer_call_fn_6076inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_6_layer_call_and_return_conditional_losses_6107inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
$__inference_dense_layer_call_fn_6256�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
?__inference_dense_layer_call_and_return_conditional_losses_6287�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_1_layer_call_fn_6296�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_1_layer_call_and_return_conditional_losses_6326�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_sequential_layer_call_fn_1983dense_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_sequential_layer_call_fn_6120inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_sequential_layer_call_fn_6133inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_sequential_layer_call_fn_2056dense_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_sequential_layer_call_and_return_conditional_losses_6190inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_sequential_layer_call_and_return_conditional_losses_6247inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_sequential_layer_call_and_return_conditional_losses_2070dense_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_sequential_layer_call_and_return_conditional_losses_2084dense_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
$__inference_dense_layer_call_fn_6256inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_dense_layer_call_and_return_conditional_losses_6287inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
&__inference_dense_1_layer_call_fn_6296inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_1_layer_call_and_return_conditional_losses_6326inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
__inference__wrapped_model_1891�*}|~��������������>?MN\]klz{0�-
&�#
!�
input_1���������-
� "5�2
0
dense_6%�"
dense_6����������
A__inference_dense_1_layer_call_and_return_conditional_losses_6326h��4�1
*�'
%�"
inputs���������-�
� "*�'
 �
0���������-�
� �
&__inference_dense_1_layer_call_fn_6296[��4�1
*�'
%�"
inputs���������-�
� "����������-��
A__inference_dense_2_layer_call_and_return_conditional_losses_5839f>?4�1
*�'
%�"
inputs����������-
� "*�'
 �
0����������
� �
&__inference_dense_2_layer_call_fn_5808Y>?4�1
*�'
%�"
inputs����������-
� "������������
A__inference_dense_3_layer_call_and_return_conditional_losses_5906fMN4�1
*�'
%�"
inputs����������
� "*�'
 �
0����������
� �
&__inference_dense_3_layer_call_fn_5875YMN4�1
*�'
%�"
inputs����������
� "������������
A__inference_dense_4_layer_call_and_return_conditional_losses_5973e\]4�1
*�'
%�"
inputs����������
� ")�&
�
0���������@
� �
&__inference_dense_4_layer_call_fn_5942X\]4�1
*�'
%�"
inputs����������
� "����������@�
A__inference_dense_5_layer_call_and_return_conditional_losses_6040dkl3�0
)�&
$�!
inputs���������@
� ")�&
�
0���������
� �
&__inference_dense_5_layer_call_fn_6009Wkl3�0
)�&
$�!
inputs���������@
� "�����������
A__inference_dense_6_layer_call_and_return_conditional_losses_6107dz{3�0
)�&
$�!
inputs���������
� ")�&
�
0���������
� �
&__inference_dense_6_layer_call_fn_6076Wz{3�0
)�&
$�!
inputs���������
� "�����������
?__inference_dense_layer_call_and_return_conditional_losses_6287h��4�1
*�'
%�"
inputs���������-�
� "*�'
 �
0���������-�
� �
$__inference_dense_layer_call_fn_6256[��4�1
*�'
%�"
inputs���������-�
� "����������-��
C__inference_dropout_2_layer_call_and_return_conditional_losses_5787f8�5
.�+
%�"
inputs����������-
p 
� "*�'
 �
0����������-
� �
C__inference_dropout_2_layer_call_and_return_conditional_losses_5799f8�5
.�+
%�"
inputs����������-
p
� "*�'
 �
0����������-
� �
(__inference_dropout_2_layer_call_fn_5777Y8�5
.�+
%�"
inputs����������-
p 
� "�����������-�
(__inference_dropout_2_layer_call_fn_5782Y8�5
.�+
%�"
inputs����������-
p
� "�����������-�
C__inference_dropout_3_layer_call_and_return_conditional_losses_5854f8�5
.�+
%�"
inputs����������
p 
� "*�'
 �
0����������
� �
C__inference_dropout_3_layer_call_and_return_conditional_losses_5866f8�5
.�+
%�"
inputs����������
p
� "*�'
 �
0����������
� �
(__inference_dropout_3_layer_call_fn_5844Y8�5
.�+
%�"
inputs����������
p 
� "������������
(__inference_dropout_3_layer_call_fn_5849Y8�5
.�+
%�"
inputs����������
p
� "������������
C__inference_dropout_4_layer_call_and_return_conditional_losses_5921f8�5
.�+
%�"
inputs����������
p 
� "*�'
 �
0����������
� �
C__inference_dropout_4_layer_call_and_return_conditional_losses_5933f8�5
.�+
%�"
inputs����������
p
� "*�'
 �
0����������
� �
(__inference_dropout_4_layer_call_fn_5911Y8�5
.�+
%�"
inputs����������
p 
� "������������
(__inference_dropout_4_layer_call_fn_5916Y8�5
.�+
%�"
inputs����������
p
� "������������
C__inference_dropout_5_layer_call_and_return_conditional_losses_5988d7�4
-�*
$�!
inputs���������@
p 
� ")�&
�
0���������@
� �
C__inference_dropout_5_layer_call_and_return_conditional_losses_6000d7�4
-�*
$�!
inputs���������@
p
� ")�&
�
0���������@
� �
(__inference_dropout_5_layer_call_fn_5978W7�4
-�*
$�!
inputs���������@
p 
� "����������@�
(__inference_dropout_5_layer_call_fn_5983W7�4
-�*
$�!
inputs���������@
p
� "����������@�
C__inference_dropout_6_layer_call_and_return_conditional_losses_6055d7�4
-�*
$�!
inputs���������
p 
� ")�&
�
0���������
� �
C__inference_dropout_6_layer_call_and_return_conditional_losses_6067d7�4
-�*
$�!
inputs���������
p
� ")�&
�
0���������
� �
(__inference_dropout_6_layer_call_fn_6045W7�4
-�*
$�!
inputs���������
p 
� "�����������
(__inference_dropout_6_layer_call_fn_6050W7�4
-�*
$�!
inputs���������
p
� "�����������
?__inference_model_layer_call_and_return_conditional_losses_3506�*}|~��������������>?MN\]klz{8�5
.�+
!�
input_1���������-
p 

 
� ")�&
�
0���������
� �
?__inference_model_layer_call_and_return_conditional_losses_3647�*}|~��������������>?MN\]klz{8�5
.�+
!�
input_1���������-
p

 
� ")�&
�
0���������
� �
?__inference_model_layer_call_and_return_conditional_losses_4556�*}|~��������������>?MN\]klz{7�4
-�*
 �
inputs���������-
p 

 
� ")�&
�
0���������
� �
?__inference_model_layer_call_and_return_conditional_losses_5380�*}|~��������������>?MN\]klz{7�4
-�*
 �
inputs���������-
p

 
� ")�&
�
0���������
� �
$__inference_model_layer_call_fn_2645�*}|~��������������>?MN\]klz{8�5
.�+
!�
input_1���������-
p 

 
� "�����������
$__inference_model_layer_call_fn_3365�*}|~��������������>?MN\]klz{8�5
.�+
!�
input_1���������-
p

 
� "�����������
$__inference_model_layer_call_fn_3771�*}|~��������������>?MN\]klz{7�4
-�*
 �
inputs���������-
p 

 
� "�����������
$__inference_model_layer_call_fn_3832�*}|~��������������>?MN\]klz{7�4
-�*
 �
inputs���������-
p

 
� "�����������
A__inference_reshape_layer_call_and_return_conditional_losses_5772b4�1
*�'
%�"
inputs���������-�
� "*�'
 �
0����������-
� 
&__inference_reshape_layer_call_fn_5759U4�1
*�'
%�"
inputs���������-�
� "�����������-�
D__inference_sequential_layer_call_and_return_conditional_losses_2070y����A�>
7�4
*�'
dense_input���������-�
p 

 
� "*�'
 �
0���������-�
� �
D__inference_sequential_layer_call_and_return_conditional_losses_2084y����A�>
7�4
*�'
dense_input���������-�
p

 
� "*�'
 �
0���������-�
� �
D__inference_sequential_layer_call_and_return_conditional_losses_6190t����<�9
2�/
%�"
inputs���������-�
p 

 
� "*�'
 �
0���������-�
� �
D__inference_sequential_layer_call_and_return_conditional_losses_6247t����<�9
2�/
%�"
inputs���������-�
p

 
� "*�'
 �
0���������-�
� �
)__inference_sequential_layer_call_fn_1983l����A�>
7�4
*�'
dense_input���������-�
p 

 
� "����������-��
)__inference_sequential_layer_call_fn_2056l����A�>
7�4
*�'
dense_input���������-�
p

 
� "����������-��
)__inference_sequential_layer_call_fn_6120g����<�9
2�/
%�"
inputs���������-�
p 

 
� "����������-��
)__inference_sequential_layer_call_fn_6133g����<�9
2�/
%�"
inputs���������-�
p

 
� "����������-��
"__inference_signature_wrapper_3710�*}|~��������������>?MN\]klz{;�8
� 
1�.
,
input_1!�
input_1���������-"5�2
0
dense_6%�"
dense_6����������
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_5413\}|*�'
 �
�
x���������-
� "*�'
 �
0���������-�
� �
;__inference_token_and_position_embedding_layer_call_fn_5389O}|*�'
 �
�
x���������-
� "����������-��
K__inference_transformer_block_layer_call_and_return_conditional_losses_5614�~��������������8�5
.�+
%�"
inputs���������-�
p 
� "*�'
 �
0���������-�
� �
K__inference_transformer_block_layer_call_and_return_conditional_losses_5754�~��������������8�5
.�+
%�"
inputs���������-�
p
� "*�'
 �
0���������-�
� �
0__inference_transformer_block_layer_call_fn_5450y~��������������8�5
.�+
%�"
inputs���������-�
p 
� "����������-��
0__inference_transformer_block_layer_call_fn_5487y~��������������8�5
.�+
%�"
inputs���������-�
p
� "����������-�