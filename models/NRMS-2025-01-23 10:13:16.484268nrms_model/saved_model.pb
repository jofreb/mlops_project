��/
��
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
p
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2	"
adj_xbool( "
adj_ybool( 
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
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
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
d
Shape

input"T&
output"out_type��out_type"	
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
-
Sqrt
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
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
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628��+
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:�*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:�*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:�*
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:�*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
�
Adam/v/att_layer2/qVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameAdam/v/att_layer2/q
|
'Adam/v/att_layer2/q/Read/ReadVariableOpReadVariableOpAdam/v/att_layer2/q*
_output_shapes
:	�*
dtype0
�
Adam/m/att_layer2/qVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameAdam/m/att_layer2/q
|
'Adam/m/att_layer2/q/Read/ReadVariableOpReadVariableOpAdam/m/att_layer2/q*
_output_shapes
:	�*
dtype0

Adam/v/att_layer2/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/v/att_layer2/b
x
'Adam/v/att_layer2/b/Read/ReadVariableOpReadVariableOpAdam/v/att_layer2/b*
_output_shapes	
:�*
dtype0

Adam/m/att_layer2/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/m/att_layer2/b
x
'Adam/m/att_layer2/b/Read/ReadVariableOpReadVariableOpAdam/m/att_layer2/b*
_output_shapes	
:�*
dtype0
�
Adam/v/att_layer2/WVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*$
shared_nameAdam/v/att_layer2/W
}
'Adam/v/att_layer2/W/Read/ReadVariableOpReadVariableOpAdam/v/att_layer2/W* 
_output_shapes
:
��*
dtype0
�
Adam/m/att_layer2/WVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*$
shared_nameAdam/m/att_layer2/W
}
'Adam/m/att_layer2/W/Read/ReadVariableOpReadVariableOpAdam/m/att_layer2/W* 
_output_shapes
:
��*
dtype0
�
Adam/v/self_attention/WVVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameAdam/v/self_attention/WV
�
,Adam/v/self_attention/WV/Read/ReadVariableOpReadVariableOpAdam/v/self_attention/WV* 
_output_shapes
:
��*
dtype0
�
Adam/m/self_attention/WVVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameAdam/m/self_attention/WV
�
,Adam/m/self_attention/WV/Read/ReadVariableOpReadVariableOpAdam/m/self_attention/WV* 
_output_shapes
:
��*
dtype0
�
Adam/v/self_attention/WKVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameAdam/v/self_attention/WK
�
,Adam/v/self_attention/WK/Read/ReadVariableOpReadVariableOpAdam/v/self_attention/WK* 
_output_shapes
:
��*
dtype0
�
Adam/m/self_attention/WKVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameAdam/m/self_attention/WK
�
,Adam/m/self_attention/WK/Read/ReadVariableOpReadVariableOpAdam/m/self_attention/WK* 
_output_shapes
:
��*
dtype0
�
Adam/v/self_attention/WQVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameAdam/v/self_attention/WQ
�
,Adam/v/self_attention/WQ/Read/ReadVariableOpReadVariableOpAdam/v/self_attention/WQ* 
_output_shapes
:
��*
dtype0
�
Adam/m/self_attention/WQVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameAdam/m/self_attention/WQ
�
,Adam/m/self_attention/WQ/Read/ReadVariableOpReadVariableOpAdam/m/self_attention/WQ* 
_output_shapes
:
��*
dtype0

Adam/v/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/v/dense_3/bias
x
'Adam/v/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/bias*
_output_shapes	
:�*
dtype0

Adam/m/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/m/dense_3/bias
x
'Adam/m/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/v/dense_3/kernel
�
)Adam/v/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/m/dense_3/kernel
�
)Adam/m/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/kernel* 
_output_shapes
:
��*
dtype0
�
!Adam/v/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/v/batch_normalization_2/beta
�
5Adam/v/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
!Adam/m/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/m/batch_normalization_2/beta
�
5Adam/m/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
"Adam/v/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/v/batch_normalization_2/gamma
�
6Adam/v/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_2/gamma*
_output_shapes	
:�*
dtype0
�
"Adam/m/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/m/batch_normalization_2/gamma
�
6Adam/m/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_2/gamma*
_output_shapes	
:�*
dtype0

Adam/v/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/v/dense_2/bias
x
'Adam/v/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/bias*
_output_shapes	
:�*
dtype0

Adam/m/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/m/dense_2/bias
x
'Adam/m/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/v/dense_2/kernel
�
)Adam/v/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/m/dense_2/kernel
�
)Adam/m/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
!Adam/v/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/v/batch_normalization_1/beta
�
5Adam/v/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
!Adam/m/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/m/batch_normalization_1/beta
�
5Adam/m/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
"Adam/v/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/v/batch_normalization_1/gamma
�
6Adam/v/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_1/gamma*
_output_shapes	
:�*
dtype0
�
"Adam/m/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/m/batch_normalization_1/gamma
�
6Adam/m/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_1/gamma*
_output_shapes	
:�*
dtype0

Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/v/dense_1/bias
x
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes	
:�*
dtype0

Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/m/dense_1/bias
x
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/v/dense_1/kernel
�
)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/m/dense_1/kernel
�
)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!Adam/v/batch_normalization/beta
�
3Adam/v/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/v/batch_normalization/beta*
_output_shapes	
:�*
dtype0
�
Adam/m/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!Adam/m/batch_normalization/beta
�
3Adam/m/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/m/batch_normalization/beta*
_output_shapes	
:�*
dtype0
�
 Adam/v/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/v/batch_normalization/gamma
�
4Adam/v/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/v/batch_normalization/gamma*
_output_shapes	
:�*
dtype0
�
 Adam/m/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/m/batch_normalization/gamma
�
4Adam/m/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/m/batch_normalization/gamma*
_output_shapes	
:�*
dtype0
{
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/v/dense/bias
t
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes	
:�*
dtype0
{
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/m/dense/bias
t
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*$
shared_nameAdam/v/dense/kernel
}
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*$
shared_nameAdam/m/dense/kernel
}
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel* 
_output_shapes
:
��*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
u
att_layer2/qVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_nameatt_layer2/q
n
 att_layer2/q/Read/ReadVariableOpReadVariableOpatt_layer2/q*
_output_shapes
:	�*
dtype0
q
att_layer2/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameatt_layer2/b
j
 att_layer2/b/Read/ReadVariableOpReadVariableOpatt_layer2/b*
_output_shapes	
:�*
dtype0
v
att_layer2/WVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_nameatt_layer2/W
o
 att_layer2/W/Read/ReadVariableOpReadVariableOpatt_layer2/W* 
_output_shapes
:
��*
dtype0
�
self_attention/WVVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*"
shared_nameself_attention/WV
y
%self_attention/WV/Read/ReadVariableOpReadVariableOpself_attention/WV* 
_output_shapes
:
��*
dtype0
�
self_attention/WKVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*"
shared_nameself_attention/WK
y
%self_attention/WK/Read/ReadVariableOpReadVariableOpself_attention/WK* 
_output_shapes
:
��*
dtype0
�
self_attention/WQVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*"
shared_nameself_attention/WQ
y
%self_attention/WQ/Read/ReadVariableOpReadVariableOpself_attention/WQ* 
_output_shapes
:
��*
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_2/moving_variance
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:�*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:�*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:�*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:�*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:�*
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
��*
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:�*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:�*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
��*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:�*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
��*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:�*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
��*
dtype0
�
serving_default_input_1Placeholder*,
_output_shapes
:���������#�*
dtype0*!
shape:���������#�
�
serving_default_input_2Placeholder*5
_output_shapes#
!:�������������������*
dtype0**
shape!:�������������������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2dense/kernel
dense/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense_1/kerneldense_1/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betadense_2/kerneldense_2/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betadense_3/kerneldense_3/biasself_attention/WQself_attention/WKself_attention/WVatt_layer2/Watt_layer2/batt_layer2/q*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������*<
_read_only_resource_inputs
	
*<
config_proto,*

CPU

GPU 2J 8R(���������� *,
f'R%
#__inference_signature_wrapper_85206

NoOpNoOp
ث
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	layer*
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses*
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses* 
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses* 
�
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
;14
<15
=16
>17
?18
@19
A20
B21
C22
D23
E24
F25*
�
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
A14
B15
C16
D17
E18
F19*
* 
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ltrace_0
Mtrace_1* 

Ntrace_0
Otrace_1* 
* 
�
P
_variables
Q_iterations
R_learning_rate
S_index_dict
T
_momentums
U_velocities
V_update_step_xla*

Wserving_default* 
�
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
;14
<15
=16
>17
?18
@19*
j
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13*
* 
�
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

]trace_0
^trace_1* 

_trace_0
`trace_1* 
�
alayer-0
blayer_with_weights-0
blayer-1
clayer_with_weights-1
clayer-2
dlayer-3
elayer_with_weights-2
elayer-4
flayer_with_weights-3
flayer-5
glayer-6
hlayer_with_weights-4
hlayer-7
ilayer_with_weights-5
ilayer-8
jlayer-9
klayer_with_weights-6
klayer-10
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses*
* 
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
	layer*
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses
AWQ
BWK
CWV*
�
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
DW
Eb
Fq*
�
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
;14
<15
=16
>17
?18
@19
A20
B21
C22
D23
E24
F25*
�
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
A14
B15
C16
D17
E18
F19*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses* 
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
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
LF
VARIABLE_VALUEdense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEbatch_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEbatch_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_1/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbatch_normalization_1/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_2/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_2/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_2/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_2/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEbatch_normalization/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#batch_normalization/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!batch_normalization_2/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%batch_normalization_2/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEself_attention/WQ'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEself_attention/WK'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEself_attention/WV'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEatt_layer2/W'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEatt_layer2/b'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEatt_layer2/q'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
.
;0
<1
=2
>3
?4
@5*
.
0
1
2
3
4
5*

�0
�1*
* 
* 
* 
* 
* 
* 
�
Q0
�1
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
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
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
�15
�16
�17
�18
�19*
�
�0
�1
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
�15
�16
�17
�18
�19*
* 
* 
.
;0
<1
=2
>3
?4
@5*

0*
* 
* 
* 
* 
* 
* 
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

-kernel
.bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	/gamma
0beta
;moving_mean
<moving_variance*
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

1kernel
2bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	3gamma
4beta
=moving_mean
>moving_variance*
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

5kernel
6bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	7gamma
8beta
?moving_mean
@moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

9kernel
:bias*
�
-0
.1
/2
03
;4
<5
16
27
38
49
=10
>11
512
613
714
815
?16
@17
918
:19*
j
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
;14
<15
=16
>17
?18
@19*
j
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 

A0
B1
C2*

A0
B1
C2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

D0
E1
F2*

D0
E1
F2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
.
;0
<1
=2
>3
?4
@5*
 
0
1
2
3*
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
<
�	variables
�	keras_api

�total

�count*
z
�	variables
�	keras_api
�true_positives
�true_negatives
�false_positives
�false_negatives*
^X
VARIABLE_VALUEAdam/m/dense/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/dense/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/dense/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/batch_normalization/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/batch_normalization/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/batch_normalization/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/batch_normalization/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_1/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_1/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_1/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_1/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_1/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_1/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_1/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_1/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_2/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_2/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_2/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_2/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_2/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_2/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_2/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_2/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_3/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_3/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_3/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_3/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/self_attention/WQ2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/self_attention/WQ2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/self_attention/WK2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/self_attention/WK2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/self_attention/WV2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/self_attention/WV2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/att_layer2/W2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/att_layer2/W2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/att_layer2/b2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/att_layer2/b2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/att_layer2/q2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/att_layer2/q2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*

-0
.1*

-0
.1*
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

�trace_0* 

�trace_0* 
 
/0
01
;2
<3*

/0
01*
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

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
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
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

10
21*

10
21*
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

�trace_0* 

�trace_0* 
 
30
41
=2
>3*

30
41*
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

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
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
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

50
61*

50
61*
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

�trace_0* 

�trace_0* 
 
70
81
?2
@3*

70
81*
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

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
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
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

90
:1*

90
:1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
.
;0
<1
=2
>3
?4
@5*
R
a0
b1
c2
d3
e4
f5
g6
h7
i8
j9
k10*
* 
* 
* 
* 
* 
* 
* 
.
;0
<1
=2
>3
?4
@5*

0*
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

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�	variables*
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 

;0
<1*
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

=0
>1*
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

?0
@1*
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
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasbatch_normalization/gammabatch_normalization/betadense_1/kerneldense_1/biasbatch_normalization_1/gammabatch_normalization_1/betadense_2/kerneldense_2/biasbatch_normalization_2/gammabatch_normalization_2/betadense_3/kerneldense_3/biasbatch_normalization/moving_mean#batch_normalization/moving_variance!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceself_attention/WQself_attention/WKself_attention/WVatt_layer2/Watt_layer2/batt_layer2/q	iterationlearning_rateAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/bias Adam/m/batch_normalization/gamma Adam/v/batch_normalization/gammaAdam/m/batch_normalization/betaAdam/v/batch_normalization/betaAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/bias"Adam/m/batch_normalization_1/gamma"Adam/v/batch_normalization_1/gamma!Adam/m/batch_normalization_1/beta!Adam/v/batch_normalization_1/betaAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/bias"Adam/m/batch_normalization_2/gamma"Adam/v/batch_normalization_2/gamma!Adam/m/batch_normalization_2/beta!Adam/v/batch_normalization_2/betaAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biasAdam/m/self_attention/WQAdam/v/self_attention/WQAdam/m/self_attention/WKAdam/v/self_attention/WKAdam/m/self_attention/WVAdam/v/self_attention/WVAdam/m/att_layer2/WAdam/v/att_layer2/WAdam/m/att_layer2/bAdam/v/att_layer2/bAdam/m/att_layer2/qAdam/v/att_layer2/qtotalcounttrue_positivestrue_negativesfalse_positivesfalse_negativesConst*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *'
f"R 
__inference__traced_save_87070
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasbatch_normalization/gammabatch_normalization/betadense_1/kerneldense_1/biasbatch_normalization_1/gammabatch_normalization_1/betadense_2/kerneldense_2/biasbatch_normalization_2/gammabatch_normalization_2/betadense_3/kerneldense_3/biasbatch_normalization/moving_mean#batch_normalization/moving_variance!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceself_attention/WQself_attention/WKself_attention/WVatt_layer2/Watt_layer2/batt_layer2/q	iterationlearning_rateAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/bias Adam/m/batch_normalization/gamma Adam/v/batch_normalization/gammaAdam/m/batch_normalization/betaAdam/v/batch_normalization/betaAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/bias"Adam/m/batch_normalization_1/gamma"Adam/v/batch_normalization_1/gamma!Adam/m/batch_normalization_1/beta!Adam/v/batch_normalization_1/betaAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/bias"Adam/m/batch_normalization_2/gamma"Adam/v/batch_normalization_2/gamma!Adam/m/batch_normalization_2/beta!Adam/v/batch_normalization_2/betaAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biasAdam/m/self_attention/WQAdam/v/self_attention/WQAdam/m/self_attention/WKAdam/v/self_attention/WKAdam/m/self_attention/WVAdam/v/self_attention/WVAdam/m/att_layer2/WAdam/v/att_layer2/WAdam/m/att_layer2/bAdam/v/att_layer2/bAdam/m/att_layer2/qAdam/v/att_layer2/qtotalcounttrue_positivestrue_negativesfalse_positivesfalse_negatives*V
TinO
M2K*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� **
f%R#
!__inference__traced_restore_87301��(
�
a
E__inference_activation_layer_call_and_return_conditional_losses_84873

inputs
identityU
SoftmaxSoftmaxinputs*
T0*0
_output_shapes
:������������������b
IdentityIdentitySoftmax:softmax:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:������������������:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
E
)__inference_dropout_2_layer_call_fn_86566

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_83383a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�/
�
E__inference_att_layer2_layer_call_and_return_conditional_losses_86202

inputs3
shape_1_readvariableop_resource:
��*
add_readvariableop_resource:	�2
shape_3_readvariableop_resource:	�
identity��add/ReadVariableOp�transpose/ReadVariableOp�transpose_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"�  �   S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������z
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       |
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0* 
_output_shapes
:
��`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"�  ����h
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
��i
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*(
_output_shapes
:����������S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#T
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:w
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*,
_output_shapes
:���������#�k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0s
addAddV2Reshape_2:output:0add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������#�L
TanhTanhadd:z:0*
T0*,
_output_shapes
:���������#�M
Shape_2ShapeTanh:y:0*
T0*
_output_shapes
::��U
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numw
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"�      S
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   k
	Reshape_3ReshapeTanh:y:0Reshape_3/shape:output:0*
T0*(
_output_shapes
:����������{
transpose_1/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes
:	�*
dtype0a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       �
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes
:	�`
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"�   ����i
	Reshape_4Reshapetranspose_1:y:0Reshape_4/shape:output:0*
T0*
_output_shapes
:	�l
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:���������S
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#S
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:x
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:���������#o
SqueezeSqueezeReshape_5:output:0*
T0*'
_output_shapes
:���������#*
squeeze_dims
N
ExpExpSqueeze:output:0*
T0*'
_output_shapes
:���������#`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������v
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3`
add_1AddV2Sum:output:0add_1/y:output:0*
T0*'
_output_shapes
:���������X
truedivRealDivExp:y:0	add_1:z:0*
T0*'
_output_shapes
:���������#Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������t

ExpandDims
ExpandDimstruediv:z:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������#^
mulMulinputsExpandDims:output:0*
T0*,
_output_shapes
:���������#�Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :j
Sum_1Summul:z:0 Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:����������^
IdentityIdentitySum_1:output:0^NoOp*
T0*(
_output_shapes
:����������o
NoOpNoOp^add/ReadVariableOp^transpose/ReadVariableOp^transpose_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:���������#�: : : 2(
add/ReadVariableOpadd/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
,
_output_shapes
:���������#�
 
_user_specified_nameinputs
�&
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_86536

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�&
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_86282

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�!
�
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_83659

inputs&
news_encoder_83613:
��!
news_encoder_83615:	�!
news_encoder_83617:	�!
news_encoder_83619:	�!
news_encoder_83621:	�!
news_encoder_83623:	�&
news_encoder_83625:
��!
news_encoder_83627:	�!
news_encoder_83629:	�!
news_encoder_83631:	�!
news_encoder_83633:	�!
news_encoder_83635:	�&
news_encoder_83637:
��!
news_encoder_83639:	�!
news_encoder_83641:	�!
news_encoder_83643:	�!
news_encoder_83645:	�!
news_encoder_83647:	�&
news_encoder_83649:
��!
news_encoder_83651:	�
identity��$news_encoder/StatefulPartitionedCallI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
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
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:�����������
$news_encoder/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0news_encoder_83613news_encoder_83615news_encoder_83617news_encoder_83619news_encoder_83621news_encoder_83623news_encoder_83625news_encoder_83627news_encoder_83629news_encoder_83631news_encoder_83633news_encoder_83635news_encoder_83637news_encoder_83639news_encoder_83641news_encoder_83643news_encoder_83645news_encoder_83647news_encoder_83649news_encoder_83651* 
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*6
_read_only_resource_inputs
	
*<
config_proto,*

CPU

GPU 2J 8R(���������� *P
fKRI
G__inference_news_encoder_layer_call_and_return_conditional_losses_83391\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape-news_encoder/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������I
NoOpNoOp%^news_encoder/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�������������������: : : : : : : : : : : : : : : : : : : : 2L
$news_encoder/StatefulPartitionedCall$news_encoder/StatefulPartitionedCall:%!

_user_specified_name83651:%!

_user_specified_name83649:%!

_user_specified_name83647:%!

_user_specified_name83645:%!

_user_specified_name83643:%!

_user_specified_name83641:%!

_user_specified_name83639:%!

_user_specified_name83637:%!

_user_specified_name83635:%!

_user_specified_name83633:%
!

_user_specified_name83631:%	!

_user_specified_name83629:%!

_user_specified_name83627:%!

_user_specified_name83625:%!

_user_specified_name83623:%!

_user_specified_name83621:%!

_user_specified_name83619:%!

_user_specified_name83617:%!

_user_specified_name83615:%!

_user_specified_name83613:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_85206
input_1
input_2
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:
��

unknown_20:
��

unknown_21:
��

unknown_22:
��

unknown_23:	�

unknown_24:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������*<
_read_only_resource_inputs
	
*<
config_proto,*

CPU

GPU 2J 8R(���������� *)
f$R"
 __inference__wrapped_model_82948x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:���������#�:�������������������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name85202:%!

_user_specified_name85200:%!

_user_specified_name85198:%!

_user_specified_name85196:%!

_user_specified_name85194:%!

_user_specified_name85192:%!

_user_specified_name85190:%!

_user_specified_name85188:%!

_user_specified_name85186:%!

_user_specified_name85184:%!

_user_specified_name85182:%!

_user_specified_name85180:%!

_user_specified_name85178:%!

_user_specified_name85176:%!

_user_specified_name85174:%!

_user_specified_name85172:%!

_user_specified_name85170:%
!

_user_specified_name85168:%	!

_user_specified_name85166:%!

_user_specified_name85164:%!

_user_specified_name85162:%!

_user_specified_name85160:%!

_user_specified_name85158:%!

_user_specified_name85156:%!

_user_specified_name85154:%!

_user_specified_name85152:^Z
5
_output_shapes#
!:�������������������
!
_user_specified_name	input_2:U Q
,
_output_shapes
:���������#�
!
_user_specified_name	input_1
�

�
B__inference_dense_2_layer_call_and_return_conditional_losses_83277

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�/
�
E__inference_att_layer2_layer_call_and_return_conditional_losses_86140

inputs3
shape_1_readvariableop_resource:
��*
add_readvariableop_resource:	�2
shape_3_readvariableop_resource:	�
identity��add/ReadVariableOp�transpose/ReadVariableOp�transpose_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"�  �   S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������z
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       |
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0* 
_output_shapes
:
��`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"�  ����h
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
��i
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*(
_output_shapes
:����������S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#T
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:w
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*,
_output_shapes
:���������#�k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0s
addAddV2Reshape_2:output:0add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������#�L
TanhTanhadd:z:0*
T0*,
_output_shapes
:���������#�M
Shape_2ShapeTanh:y:0*
T0*
_output_shapes
::��U
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numw
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"�      S
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   k
	Reshape_3ReshapeTanh:y:0Reshape_3/shape:output:0*
T0*(
_output_shapes
:����������{
transpose_1/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes
:	�*
dtype0a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       �
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes
:	�`
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"�   ����i
	Reshape_4Reshapetranspose_1:y:0Reshape_4/shape:output:0*
T0*
_output_shapes
:	�l
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:���������S
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#S
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:x
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:���������#o
SqueezeSqueezeReshape_5:output:0*
T0*'
_output_shapes
:���������#*
squeeze_dims
N
ExpExpSqueeze:output:0*
T0*'
_output_shapes
:���������#`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������v
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3`
add_1AddV2Sum:output:0add_1/y:output:0*
T0*'
_output_shapes
:���������X
truedivRealDivExp:y:0	add_1:z:0*
T0*'
_output_shapes
:���������#Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������t

ExpandDims
ExpandDimstruediv:z:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������#^
mulMulinputsExpandDims:output:0*
T0*,
_output_shapes
:���������#�Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :j
Sum_1Summul:z:0 Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:����������^
IdentityIdentitySum_1:output:0^NoOp*
T0*(
_output_shapes
:����������o
NoOpNoOp^add/ReadVariableOp^transpose/ReadVariableOp^transpose_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:���������#�: : : 2(
add/ReadVariableOpadd/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
,
_output_shapes
:���������#�
 
_user_specified_nameinputs
��
�
G__inference_user_encoder_layer_call_and_return_conditional_losses_84507
input_5*
time_distributed_84328:
��%
time_distributed_84330:	�%
time_distributed_84332:	�%
time_distributed_84334:	�%
time_distributed_84336:	�%
time_distributed_84338:	�*
time_distributed_84340:
��%
time_distributed_84342:	�%
time_distributed_84344:	�%
time_distributed_84346:	�%
time_distributed_84348:	�%
time_distributed_84350:	�*
time_distributed_84352:
��%
time_distributed_84354:	�%
time_distributed_84356:	�%
time_distributed_84358:	�%
time_distributed_84360:	�%
time_distributed_84362:	�*
time_distributed_84364:
��%
time_distributed_84366:	�(
self_attention_84430:
��(
self_attention_84432:
��(
self_attention_84434:
��$
att_layer2_84499:
��
att_layer2_84501:	�#
att_layer2_84503:	�
identity��"att_layer2/StatefulPartitionedCall�&self_attention/StatefulPartitionedCall�(time_distributed/StatefulPartitionedCall�=time_distributed/batch_normalization/batchnorm/ReadVariableOp�?time_distributed/batch_normalization/batchnorm/ReadVariableOp_1�?time_distributed/batch_normalization/batchnorm/ReadVariableOp_2�Atime_distributed/batch_normalization/batchnorm/mul/ReadVariableOp�?time_distributed/batch_normalization_1/batchnorm/ReadVariableOp�Atime_distributed/batch_normalization_1/batchnorm/ReadVariableOp_1�Atime_distributed/batch_normalization_1/batchnorm/ReadVariableOp_2�Ctime_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOp�?time_distributed/batch_normalization_2/batchnorm/ReadVariableOp�Atime_distributed/batch_normalization_2/batchnorm/ReadVariableOp_1�Atime_distributed/batch_normalization_2/batchnorm/ReadVariableOp_2�Ctime_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOp�-time_distributed/dense/BiasAdd/ReadVariableOp�,time_distributed/dense/MatMul/ReadVariableOp�/time_distributed/dense_1/BiasAdd/ReadVariableOp�.time_distributed/dense_1/MatMul/ReadVariableOp�/time_distributed/dense_2/BiasAdd/ReadVariableOp�.time_distributed/dense_2/MatMul/ReadVariableOp�/time_distributed/dense_3/BiasAdd/ReadVariableOp�.time_distributed/dense_3/MatMul/ReadVariableOp�
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinput_5time_distributed_84328time_distributed_84330time_distributed_84332time_distributed_84334time_distributed_84336time_distributed_84338time_distributed_84340time_distributed_84342time_distributed_84344time_distributed_84346time_distributed_84348time_distributed_84350time_distributed_84352time_distributed_84354time_distributed_84356time_distributed_84358time_distributed_84360time_distributed_84362time_distributed_84364time_distributed_84366* 
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������#�*6
_read_only_resource_inputs
	
*<
config_proto,*

CPU

GPU 2J 8R(���������� *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_83861o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
time_distributed/ReshapeReshapeinput_5'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
,time_distributed/dense/MatMul/ReadVariableOpReadVariableOptime_distributed_84328* 
_output_shapes
:
��*
dtype0�
time_distributed/dense/MatMulMatMul!time_distributed/Reshape:output:04time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOptime_distributed_84330*
_output_shapes	
:�*
dtype0�
time_distributed/dense/BiasAddBiasAdd'time_distributed/dense/MatMul:product:05time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
time_distributed/dense/ReluRelu'time_distributed/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
=time_distributed/batch_normalization/batchnorm/ReadVariableOpReadVariableOptime_distributed_84332*
_output_shapes	
:�*
dtype0y
4time_distributed/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
2time_distributed/batch_normalization/batchnorm/addAddV2Etime_distributed/batch_normalization/batchnorm/ReadVariableOp:value:0=time_distributed/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
4time_distributed/batch_normalization/batchnorm/RsqrtRsqrt6time_distributed/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Atime_distributed/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOptime_distributed_84334*
_output_shapes	
:�*
dtype0�
2time_distributed/batch_normalization/batchnorm/mulMul8time_distributed/batch_normalization/batchnorm/Rsqrt:y:0Itime_distributed/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
4time_distributed/batch_normalization/batchnorm/mul_1Mul)time_distributed/dense/Relu:activations:06time_distributed/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
?time_distributed/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOptime_distributed_84336*
_output_shapes	
:�*
dtype0�
4time_distributed/batch_normalization/batchnorm/mul_2MulGtime_distributed/batch_normalization/batchnorm/ReadVariableOp_1:value:06time_distributed/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
?time_distributed/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOptime_distributed_84338*
_output_shapes	
:�*
dtype0�
2time_distributed/batch_normalization/batchnorm/subSubGtime_distributed/batch_normalization/batchnorm/ReadVariableOp_2:value:08time_distributed/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
4time_distributed/batch_normalization/batchnorm/add_1AddV28time_distributed/batch_normalization/batchnorm/mul_1:z:06time_distributed/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
!time_distributed/dropout/IdentityIdentity8time_distributed/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
.time_distributed/dense_1/MatMul/ReadVariableOpReadVariableOptime_distributed_84340* 
_output_shapes
:
��*
dtype0�
time_distributed/dense_1/MatMulMatMul*time_distributed/dropout/Identity:output:06time_distributed/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/time_distributed/dense_1/BiasAdd/ReadVariableOpReadVariableOptime_distributed_84342*
_output_shapes	
:�*
dtype0�
 time_distributed/dense_1/BiasAddBiasAdd)time_distributed/dense_1/MatMul:product:07time_distributed/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
time_distributed/dense_1/ReluRelu)time_distributed/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
?time_distributed/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOptime_distributed_84344*
_output_shapes	
:�*
dtype0{
6time_distributed/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4time_distributed/batch_normalization_1/batchnorm/addAddV2Gtime_distributed/batch_normalization_1/batchnorm/ReadVariableOp:value:0?time_distributed/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
6time_distributed/batch_normalization_1/batchnorm/RsqrtRsqrt8time_distributed/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Ctime_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOptime_distributed_84346*
_output_shapes	
:�*
dtype0�
4time_distributed/batch_normalization_1/batchnorm/mulMul:time_distributed/batch_normalization_1/batchnorm/Rsqrt:y:0Ktime_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
6time_distributed/batch_normalization_1/batchnorm/mul_1Mul+time_distributed/dense_1/Relu:activations:08time_distributed/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Atime_distributed/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOptime_distributed_84348*
_output_shapes	
:�*
dtype0�
6time_distributed/batch_normalization_1/batchnorm/mul_2MulItime_distributed/batch_normalization_1/batchnorm/ReadVariableOp_1:value:08time_distributed/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
Atime_distributed/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOptime_distributed_84350*
_output_shapes	
:�*
dtype0�
4time_distributed/batch_normalization_1/batchnorm/subSubItime_distributed/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0:time_distributed/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
6time_distributed/batch_normalization_1/batchnorm/add_1AddV2:time_distributed/batch_normalization_1/batchnorm/mul_1:z:08time_distributed/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
#time_distributed/dropout_1/IdentityIdentity:time_distributed/batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
.time_distributed/dense_2/MatMul/ReadVariableOpReadVariableOptime_distributed_84352* 
_output_shapes
:
��*
dtype0�
time_distributed/dense_2/MatMulMatMul,time_distributed/dropout_1/Identity:output:06time_distributed/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/time_distributed/dense_2/BiasAdd/ReadVariableOpReadVariableOptime_distributed_84354*
_output_shapes	
:�*
dtype0�
 time_distributed/dense_2/BiasAddBiasAdd)time_distributed/dense_2/MatMul:product:07time_distributed/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
time_distributed/dense_2/ReluRelu)time_distributed/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
?time_distributed/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOptime_distributed_84356*
_output_shapes	
:�*
dtype0{
6time_distributed/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4time_distributed/batch_normalization_2/batchnorm/addAddV2Gtime_distributed/batch_normalization_2/batchnorm/ReadVariableOp:value:0?time_distributed/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
6time_distributed/batch_normalization_2/batchnorm/RsqrtRsqrt8time_distributed/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Ctime_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOptime_distributed_84358*
_output_shapes	
:�*
dtype0�
4time_distributed/batch_normalization_2/batchnorm/mulMul:time_distributed/batch_normalization_2/batchnorm/Rsqrt:y:0Ktime_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
6time_distributed/batch_normalization_2/batchnorm/mul_1Mul+time_distributed/dense_2/Relu:activations:08time_distributed/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Atime_distributed/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOptime_distributed_84360*
_output_shapes	
:�*
dtype0�
6time_distributed/batch_normalization_2/batchnorm/mul_2MulItime_distributed/batch_normalization_2/batchnorm/ReadVariableOp_1:value:08time_distributed/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
Atime_distributed/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOptime_distributed_84362*
_output_shapes	
:�*
dtype0�
4time_distributed/batch_normalization_2/batchnorm/subSubItime_distributed/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0:time_distributed/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
6time_distributed/batch_normalization_2/batchnorm/add_1AddV2:time_distributed/batch_normalization_2/batchnorm/mul_1:z:08time_distributed/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
#time_distributed/dropout_2/IdentityIdentity:time_distributed/batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
.time_distributed/dense_3/MatMul/ReadVariableOpReadVariableOptime_distributed_84364* 
_output_shapes
:
��*
dtype0�
time_distributed/dense_3/MatMulMatMul,time_distributed/dropout_2/Identity:output:06time_distributed/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/time_distributed/dense_3/BiasAdd/ReadVariableOpReadVariableOptime_distributed_84366*
_output_shapes	
:�*
dtype0�
 time_distributed/dense_3/BiasAddBiasAdd)time_distributed/dense_3/MatMul:product:07time_distributed/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
time_distributed/dense_3/ReluRelu)time_distributed/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
&self_attention/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:01time_distributed/StatefulPartitionedCall:output:01time_distributed/StatefulPartitionedCall:output:0self_attention_84430self_attention_84432self_attention_84434*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������#�*%
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *R
fMRK
I__inference_self_attention_layer_call_and_return_conditional_losses_84247�
"att_layer2/StatefulPartitionedCallStatefulPartitionedCall/self_attention/StatefulPartitionedCall:output:0att_layer2_84499att_layer2_84501att_layer2_84503*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *N
fIRG
E__inference_att_layer2_layer_call_and_return_conditional_losses_84498{
IdentityIdentity+att_layer2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������

NoOpNoOp#^att_layer2/StatefulPartitionedCall'^self_attention/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall>^time_distributed/batch_normalization/batchnorm/ReadVariableOp@^time_distributed/batch_normalization/batchnorm/ReadVariableOp_1@^time_distributed/batch_normalization/batchnorm/ReadVariableOp_2B^time_distributed/batch_normalization/batchnorm/mul/ReadVariableOp@^time_distributed/batch_normalization_1/batchnorm/ReadVariableOpB^time_distributed/batch_normalization_1/batchnorm/ReadVariableOp_1B^time_distributed/batch_normalization_1/batchnorm/ReadVariableOp_2D^time_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOp@^time_distributed/batch_normalization_2/batchnorm/ReadVariableOpB^time_distributed/batch_normalization_2/batchnorm/ReadVariableOp_1B^time_distributed/batch_normalization_2/batchnorm/ReadVariableOp_2D^time_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOp.^time_distributed/dense/BiasAdd/ReadVariableOp-^time_distributed/dense/MatMul/ReadVariableOp0^time_distributed/dense_1/BiasAdd/ReadVariableOp/^time_distributed/dense_1/MatMul/ReadVariableOp0^time_distributed/dense_2/BiasAdd/ReadVariableOp/^time_distributed/dense_2/MatMul/ReadVariableOp0^time_distributed/dense_3/BiasAdd/ReadVariableOp/^time_distributed/dense_3/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:���������#�: : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"att_layer2/StatefulPartitionedCall"att_layer2/StatefulPartitionedCall2P
&self_attention/StatefulPartitionedCall&self_attention/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2�
?time_distributed/batch_normalization/batchnorm/ReadVariableOp_1?time_distributed/batch_normalization/batchnorm/ReadVariableOp_12�
?time_distributed/batch_normalization/batchnorm/ReadVariableOp_2?time_distributed/batch_normalization/batchnorm/ReadVariableOp_22~
=time_distributed/batch_normalization/batchnorm/ReadVariableOp=time_distributed/batch_normalization/batchnorm/ReadVariableOp2�
Atime_distributed/batch_normalization/batchnorm/mul/ReadVariableOpAtime_distributed/batch_normalization/batchnorm/mul/ReadVariableOp2�
Atime_distributed/batch_normalization_1/batchnorm/ReadVariableOp_1Atime_distributed/batch_normalization_1/batchnorm/ReadVariableOp_12�
Atime_distributed/batch_normalization_1/batchnorm/ReadVariableOp_2Atime_distributed/batch_normalization_1/batchnorm/ReadVariableOp_22�
?time_distributed/batch_normalization_1/batchnorm/ReadVariableOp?time_distributed/batch_normalization_1/batchnorm/ReadVariableOp2�
Ctime_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOpCtime_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOp2�
Atime_distributed/batch_normalization_2/batchnorm/ReadVariableOp_1Atime_distributed/batch_normalization_2/batchnorm/ReadVariableOp_12�
Atime_distributed/batch_normalization_2/batchnorm/ReadVariableOp_2Atime_distributed/batch_normalization_2/batchnorm/ReadVariableOp_22�
?time_distributed/batch_normalization_2/batchnorm/ReadVariableOp?time_distributed/batch_normalization_2/batchnorm/ReadVariableOp2�
Ctime_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOpCtime_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOp2^
-time_distributed/dense/BiasAdd/ReadVariableOp-time_distributed/dense/BiasAdd/ReadVariableOp2\
,time_distributed/dense/MatMul/ReadVariableOp,time_distributed/dense/MatMul/ReadVariableOp2b
/time_distributed/dense_1/BiasAdd/ReadVariableOp/time_distributed/dense_1/BiasAdd/ReadVariableOp2`
.time_distributed/dense_1/MatMul/ReadVariableOp.time_distributed/dense_1/MatMul/ReadVariableOp2b
/time_distributed/dense_2/BiasAdd/ReadVariableOp/time_distributed/dense_2/BiasAdd/ReadVariableOp2`
.time_distributed/dense_2/MatMul/ReadVariableOp.time_distributed/dense_2/MatMul/ReadVariableOp2b
/time_distributed/dense_3/BiasAdd/ReadVariableOp/time_distributed/dense_3/BiasAdd/ReadVariableOp2`
.time_distributed/dense_3/MatMul/ReadVariableOp.time_distributed/dense_3/MatMul/ReadVariableOp:%!

_user_specified_name84503:%!

_user_specified_name84501:%!

_user_specified_name84499:%!

_user_specified_name84434:%!

_user_specified_name84432:%!

_user_specified_name84430:%!

_user_specified_name84366:%!

_user_specified_name84364:%!

_user_specified_name84362:%!

_user_specified_name84360:%!

_user_specified_name84358:%!

_user_specified_name84356:%!

_user_specified_name84354:%!

_user_specified_name84352:%!

_user_specified_name84350:%!

_user_specified_name84348:%
!

_user_specified_name84346:%	!

_user_specified_name84344:%!

_user_specified_name84342:%!

_user_specified_name84340:%!

_user_specified_name84338:%!

_user_specified_name84336:%!

_user_specified_name84334:%!

_user_specified_name84332:%!

_user_specified_name84330:%!

_user_specified_name84328:U Q
,
_output_shapes
:���������#�
!
_user_specified_name	input_5
�
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_86456

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
C
'__inference_dropout_layer_call_fn_86312

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_83343a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�&
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_82982

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
.__inference_self_attention_layer_call_fn_85927

qkvs_0

qkvs_1

qkvs_2
unknown:
��
	unknown_0:
��
	unknown_1:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqkvs_0qkvs_1qkvs_2unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������#�*%
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *R
fMRK
I__inference_self_attention_layer_call_and_return_conditional_losses_84247t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������#�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:���������#�:���������#�:���������#�: : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name85923:%!

_user_specified_name85921:%!

_user_specified_name85919:TP
,
_output_shapes
:���������#�
 
_user_specified_nameqkvs_2:TP
,
_output_shapes
:���������#�
 
_user_specified_nameqkvs_1:T P
,
_output_shapes
:���������#�
 
_user_specified_nameqkvs_0
��
�
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_85547

inputsE
1news_encoder_dense_matmul_readvariableop_resource:
��A
2news_encoder_dense_biasadd_readvariableop_resource:	�Q
Bnews_encoder_batch_normalization_batchnorm_readvariableop_resource:	�U
Fnews_encoder_batch_normalization_batchnorm_mul_readvariableop_resource:	�S
Dnews_encoder_batch_normalization_batchnorm_readvariableop_1_resource:	�S
Dnews_encoder_batch_normalization_batchnorm_readvariableop_2_resource:	�G
3news_encoder_dense_1_matmul_readvariableop_resource:
��C
4news_encoder_dense_1_biasadd_readvariableop_resource:	�S
Dnews_encoder_batch_normalization_1_batchnorm_readvariableop_resource:	�W
Hnews_encoder_batch_normalization_1_batchnorm_mul_readvariableop_resource:	�U
Fnews_encoder_batch_normalization_1_batchnorm_readvariableop_1_resource:	�U
Fnews_encoder_batch_normalization_1_batchnorm_readvariableop_2_resource:	�G
3news_encoder_dense_2_matmul_readvariableop_resource:
��C
4news_encoder_dense_2_biasadd_readvariableop_resource:	�S
Dnews_encoder_batch_normalization_2_batchnorm_readvariableop_resource:	�W
Hnews_encoder_batch_normalization_2_batchnorm_mul_readvariableop_resource:	�U
Fnews_encoder_batch_normalization_2_batchnorm_readvariableop_1_resource:	�U
Fnews_encoder_batch_normalization_2_batchnorm_readvariableop_2_resource:	�G
3news_encoder_dense_3_matmul_readvariableop_resource:
��C
4news_encoder_dense_3_biasadd_readvariableop_resource:	�
identity��9news_encoder/batch_normalization/batchnorm/ReadVariableOp�;news_encoder/batch_normalization/batchnorm/ReadVariableOp_1�;news_encoder/batch_normalization/batchnorm/ReadVariableOp_2�=news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp�;news_encoder/batch_normalization_1/batchnorm/ReadVariableOp�=news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1�=news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2�?news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp�;news_encoder/batch_normalization_2/batchnorm/ReadVariableOp�=news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1�=news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2�?news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp�)news_encoder/dense/BiasAdd/ReadVariableOp�(news_encoder/dense/MatMul/ReadVariableOp�+news_encoder/dense_1/BiasAdd/ReadVariableOp�*news_encoder/dense_1/MatMul/ReadVariableOp�+news_encoder/dense_2/BiasAdd/ReadVariableOp�*news_encoder/dense_2/MatMul/ReadVariableOp�+news_encoder/dense_3/BiasAdd/ReadVariableOp�*news_encoder/dense_3/MatMul/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
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
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:�����������
(news_encoder/dense/MatMul/ReadVariableOpReadVariableOp1news_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
news_encoder/dense/MatMulMatMulReshape:output:00news_encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
)news_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp2news_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
news_encoder/dense/BiasAddBiasAdd#news_encoder/dense/MatMul:product:01news_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������w
news_encoder/dense/ReluRelu#news_encoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
9news_encoder/batch_normalization/batchnorm/ReadVariableOpReadVariableOpBnews_encoder_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0u
0news_encoder/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
.news_encoder/batch_normalization/batchnorm/addAddV2Anews_encoder/batch_normalization/batchnorm/ReadVariableOp:value:09news_encoder/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
0news_encoder/batch_normalization/batchnorm/RsqrtRsqrt2news_encoder/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
=news_encoder/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpFnews_encoder_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.news_encoder/batch_normalization/batchnorm/mulMul4news_encoder/batch_normalization/batchnorm/Rsqrt:y:0Enews_encoder/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
0news_encoder/batch_normalization/batchnorm/mul_1Mul%news_encoder/dense/Relu:activations:02news_encoder/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
;news_encoder/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpDnews_encoder_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization/batchnorm/mul_2MulCnews_encoder/batch_normalization/batchnorm/ReadVariableOp_1:value:02news_encoder/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
;news_encoder/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpDnews_encoder_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
.news_encoder/batch_normalization/batchnorm/subSubCnews_encoder/batch_normalization/batchnorm/ReadVariableOp_2:value:04news_encoder/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
0news_encoder/batch_normalization/batchnorm/add_1AddV24news_encoder/batch_normalization/batchnorm/mul_1:z:02news_encoder/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
news_encoder/dropout/IdentityIdentity4news_encoder/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
*news_encoder/dense_1/MatMul/ReadVariableOpReadVariableOp3news_encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
news_encoder/dense_1/MatMulMatMul&news_encoder/dropout/Identity:output:02news_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+news_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp4news_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
news_encoder/dense_1/BiasAddBiasAdd%news_encoder/dense_1/MatMul:product:03news_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
news_encoder/dense_1/ReluRelu%news_encoder/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
;news_encoder/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpDnews_encoder_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0w
2news_encoder/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
0news_encoder/batch_normalization_1/batchnorm/addAddV2Cnews_encoder/batch_normalization_1/batchnorm/ReadVariableOp:value:0;news_encoder/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_1/batchnorm/RsqrtRsqrt4news_encoder/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
?news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpHnews_encoder_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization_1/batchnorm/mulMul6news_encoder/batch_normalization_1/batchnorm/Rsqrt:y:0Gnews_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_1/batchnorm/mul_1Mul'news_encoder/dense_1/Relu:activations:04news_encoder/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
=news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpFnews_encoder_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
2news_encoder/batch_normalization_1/batchnorm/mul_2MulEnews_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1:value:04news_encoder/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
=news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpFnews_encoder_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization_1/batchnorm/subSubEnews_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2:value:06news_encoder/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_1/batchnorm/add_1AddV26news_encoder/batch_normalization_1/batchnorm/mul_1:z:04news_encoder/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
news_encoder/dropout_1/IdentityIdentity6news_encoder/batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
*news_encoder/dense_2/MatMul/ReadVariableOpReadVariableOp3news_encoder_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
news_encoder/dense_2/MatMulMatMul(news_encoder/dropout_1/Identity:output:02news_encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+news_encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp4news_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
news_encoder/dense_2/BiasAddBiasAdd%news_encoder/dense_2/MatMul:product:03news_encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
news_encoder/dense_2/ReluRelu%news_encoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
;news_encoder/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpDnews_encoder_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0w
2news_encoder/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
0news_encoder/batch_normalization_2/batchnorm/addAddV2Cnews_encoder/batch_normalization_2/batchnorm/ReadVariableOp:value:0;news_encoder/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_2/batchnorm/RsqrtRsqrt4news_encoder/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
?news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpHnews_encoder_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization_2/batchnorm/mulMul6news_encoder/batch_normalization_2/batchnorm/Rsqrt:y:0Gnews_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_2/batchnorm/mul_1Mul'news_encoder/dense_2/Relu:activations:04news_encoder/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
=news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpFnews_encoder_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
2news_encoder/batch_normalization_2/batchnorm/mul_2MulEnews_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1:value:04news_encoder/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
=news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpFnews_encoder_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization_2/batchnorm/subSubEnews_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2:value:06news_encoder/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_2/batchnorm/add_1AddV26news_encoder/batch_normalization_2/batchnorm/mul_1:z:04news_encoder/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
news_encoder/dropout_2/IdentityIdentity6news_encoder/batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
*news_encoder/dense_3/MatMul/ReadVariableOpReadVariableOp3news_encoder_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
news_encoder/dense_3/MatMulMatMul(news_encoder/dropout_2/Identity:output:02news_encoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+news_encoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp4news_encoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
news_encoder/dense_3/BiasAddBiasAdd%news_encoder/dense_3/MatMul:product:03news_encoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
news_encoder/dense_3/ReluRelu%news_encoder/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape'news_encoder/dense_3/Relu:activations:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:��������������������	
NoOpNoOp:^news_encoder/batch_normalization/batchnorm/ReadVariableOp<^news_encoder/batch_normalization/batchnorm/ReadVariableOp_1<^news_encoder/batch_normalization/batchnorm/ReadVariableOp_2>^news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp<^news_encoder/batch_normalization_1/batchnorm/ReadVariableOp>^news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1>^news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2@^news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp<^news_encoder/batch_normalization_2/batchnorm/ReadVariableOp>^news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1>^news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2@^news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp*^news_encoder/dense/BiasAdd/ReadVariableOp)^news_encoder/dense/MatMul/ReadVariableOp,^news_encoder/dense_1/BiasAdd/ReadVariableOp+^news_encoder/dense_1/MatMul/ReadVariableOp,^news_encoder/dense_2/BiasAdd/ReadVariableOp+^news_encoder/dense_2/MatMul/ReadVariableOp,^news_encoder/dense_3/BiasAdd/ReadVariableOp+^news_encoder/dense_3/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�������������������: : : : : : : : : : : : : : : : : : : : 2z
;news_encoder/batch_normalization/batchnorm/ReadVariableOp_1;news_encoder/batch_normalization/batchnorm/ReadVariableOp_12z
;news_encoder/batch_normalization/batchnorm/ReadVariableOp_2;news_encoder/batch_normalization/batchnorm/ReadVariableOp_22v
9news_encoder/batch_normalization/batchnorm/ReadVariableOp9news_encoder/batch_normalization/batchnorm/ReadVariableOp2~
=news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp=news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp2~
=news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1=news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_12~
=news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2=news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_22z
;news_encoder/batch_normalization_1/batchnorm/ReadVariableOp;news_encoder/batch_normalization_1/batchnorm/ReadVariableOp2�
?news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp?news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp2~
=news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1=news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_12~
=news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2=news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_22z
;news_encoder/batch_normalization_2/batchnorm/ReadVariableOp;news_encoder/batch_normalization_2/batchnorm/ReadVariableOp2�
?news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp?news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp2V
)news_encoder/dense/BiasAdd/ReadVariableOp)news_encoder/dense/BiasAdd/ReadVariableOp2T
(news_encoder/dense/MatMul/ReadVariableOp(news_encoder/dense/MatMul/ReadVariableOp2Z
+news_encoder/dense_1/BiasAdd/ReadVariableOp+news_encoder/dense_1/BiasAdd/ReadVariableOp2X
*news_encoder/dense_1/MatMul/ReadVariableOp*news_encoder/dense_1/MatMul/ReadVariableOp2Z
+news_encoder/dense_2/BiasAdd/ReadVariableOp+news_encoder/dense_2/BiasAdd/ReadVariableOp2X
*news_encoder/dense_2/MatMul/ReadVariableOp*news_encoder/dense_2/MatMul/ReadVariableOp2Z
+news_encoder/dense_3/BiasAdd/ReadVariableOp+news_encoder/dense_3/BiasAdd/ReadVariableOp2X
*news_encoder/dense_3/MatMul/ReadVariableOp*news_encoder/dense_3/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�

a
B__inference_dropout_layer_call_and_return_conditional_losses_86324

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_83303

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_dense_1_layer_call_and_return_conditional_losses_86349

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_86578

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
B__inference_dropout_layer_call_and_return_conditional_losses_86329

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_83363

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�/
�
E__inference_att_layer2_layer_call_and_return_conditional_losses_84498

inputs3
shape_1_readvariableop_resource:
��*
add_readvariableop_resource:	�2
shape_3_readvariableop_resource:	�
identity��add/ReadVariableOp�transpose/ReadVariableOp�transpose_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"�  �   S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������z
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       |
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0* 
_output_shapes
:
��`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"�  ����h
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
��i
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*(
_output_shapes
:����������S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#T
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:w
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*,
_output_shapes
:���������#�k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0s
addAddV2Reshape_2:output:0add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������#�L
TanhTanhadd:z:0*
T0*,
_output_shapes
:���������#�M
Shape_2ShapeTanh:y:0*
T0*
_output_shapes
::��U
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numw
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"�      S
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   k
	Reshape_3ReshapeTanh:y:0Reshape_3/shape:output:0*
T0*(
_output_shapes
:����������{
transpose_1/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes
:	�*
dtype0a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       �
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes
:	�`
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"�   ����i
	Reshape_4Reshapetranspose_1:y:0Reshape_4/shape:output:0*
T0*
_output_shapes
:	�l
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:���������S
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#S
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:x
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:���������#o
SqueezeSqueezeReshape_5:output:0*
T0*'
_output_shapes
:���������#*
squeeze_dims
N
ExpExpSqueeze:output:0*
T0*'
_output_shapes
:���������#`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������v
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3`
add_1AddV2Sum:output:0add_1/y:output:0*
T0*'
_output_shapes
:���������X
truedivRealDivExp:y:0	add_1:z:0*
T0*'
_output_shapes
:���������#Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������t

ExpandDims
ExpandDimstruediv:z:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������#^
mulMulinputsExpandDims:output:0*
T0*,
_output_shapes
:���������#�Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :j
Sum_1Summul:z:0 Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:����������^
IdentityIdentitySum_1:output:0^NoOp*
T0*(
_output_shapes
:����������o
NoOpNoOp^add/ReadVariableOp^transpose/ReadVariableOp^transpose_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:���������#�: : : 2(
add/ReadVariableOpadd/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
,
_output_shapes
:���������#�
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_83162

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_dense_2_layer_call_and_return_conditional_losses_86476

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
G__inference_user_encoder_layer_call_and_return_conditional_losses_84325
input_5*
time_distributed_83954:
��%
time_distributed_83956:	�%
time_distributed_83958:	�%
time_distributed_83960:	�%
time_distributed_83962:	�%
time_distributed_83964:	�*
time_distributed_83966:
��%
time_distributed_83968:	�%
time_distributed_83970:	�%
time_distributed_83972:	�%
time_distributed_83974:	�%
time_distributed_83976:	�*
time_distributed_83978:
��%
time_distributed_83980:	�%
time_distributed_83982:	�%
time_distributed_83984:	�%
time_distributed_83986:	�%
time_distributed_83988:	�*
time_distributed_83990:
��%
time_distributed_83992:	�(
self_attention_84248:
��(
self_attention_84250:
��(
self_attention_84252:
��$
att_layer2_84317:
��
att_layer2_84319:	�#
att_layer2_84321:	�
identity��"att_layer2/StatefulPartitionedCall�&self_attention/StatefulPartitionedCall�(time_distributed/StatefulPartitionedCall�4time_distributed/batch_normalization/AssignMovingAvg�Ctime_distributed/batch_normalization/AssignMovingAvg/ReadVariableOp�6time_distributed/batch_normalization/AssignMovingAvg_1�Etime_distributed/batch_normalization/AssignMovingAvg_1/ReadVariableOp�=time_distributed/batch_normalization/batchnorm/ReadVariableOp�Atime_distributed/batch_normalization/batchnorm/mul/ReadVariableOp�6time_distributed/batch_normalization_1/AssignMovingAvg�Etime_distributed/batch_normalization_1/AssignMovingAvg/ReadVariableOp�8time_distributed/batch_normalization_1/AssignMovingAvg_1�Gtime_distributed/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�?time_distributed/batch_normalization_1/batchnorm/ReadVariableOp�Ctime_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOp�6time_distributed/batch_normalization_2/AssignMovingAvg�Etime_distributed/batch_normalization_2/AssignMovingAvg/ReadVariableOp�8time_distributed/batch_normalization_2/AssignMovingAvg_1�Gtime_distributed/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�?time_distributed/batch_normalization_2/batchnorm/ReadVariableOp�Ctime_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOp�-time_distributed/dense/BiasAdd/ReadVariableOp�,time_distributed/dense/MatMul/ReadVariableOp�/time_distributed/dense_1/BiasAdd/ReadVariableOp�.time_distributed/dense_1/MatMul/ReadVariableOp�/time_distributed/dense_2/BiasAdd/ReadVariableOp�.time_distributed/dense_2/MatMul/ReadVariableOp�/time_distributed/dense_3/BiasAdd/ReadVariableOp�.time_distributed/dense_3/MatMul/ReadVariableOp�
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinput_5time_distributed_83954time_distributed_83956time_distributed_83958time_distributed_83960time_distributed_83962time_distributed_83964time_distributed_83966time_distributed_83968time_distributed_83970time_distributed_83972time_distributed_83974time_distributed_83976time_distributed_83978time_distributed_83980time_distributed_83982time_distributed_83984time_distributed_83986time_distributed_83988time_distributed_83990time_distributed_83992* 
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������#�*0
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_83805o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
time_distributed/ReshapeReshapeinput_5'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
,time_distributed/dense/MatMul/ReadVariableOpReadVariableOptime_distributed_83954* 
_output_shapes
:
��*
dtype0�
time_distributed/dense/MatMulMatMul!time_distributed/Reshape:output:04time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOptime_distributed_83956*
_output_shapes	
:�*
dtype0�
time_distributed/dense/BiasAddBiasAdd'time_distributed/dense/MatMul:product:05time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
time_distributed/dense/ReluRelu'time_distributed/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Ctime_distributed/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
1time_distributed/batch_normalization/moments/meanMean)time_distributed/dense/Relu:activations:0Ltime_distributed/batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
9time_distributed/batch_normalization/moments/StopGradientStopGradient:time_distributed/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	��
>time_distributed/batch_normalization/moments/SquaredDifferenceSquaredDifference)time_distributed/dense/Relu:activations:0Btime_distributed/batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
Gtime_distributed/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
5time_distributed/batch_normalization/moments/varianceMeanBtime_distributed/batch_normalization/moments/SquaredDifference:z:0Ptime_distributed/batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
4time_distributed/batch_normalization/moments/SqueezeSqueeze:time_distributed/batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
6time_distributed/batch_normalization/moments/Squeeze_1Squeeze>time_distributed/batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 
:time_distributed/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Ctime_distributed/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOptime_distributed_83958)^time_distributed/StatefulPartitionedCall*
_output_shapes	
:�*
dtype0�
8time_distributed/batch_normalization/AssignMovingAvg/subSubKtime_distributed/batch_normalization/AssignMovingAvg/ReadVariableOp:value:0=time_distributed/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
8time_distributed/batch_normalization/AssignMovingAvg/mulMul<time_distributed/batch_normalization/AssignMovingAvg/sub:z:0Ctime_distributed/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
4time_distributed/batch_normalization/AssignMovingAvgAssignSubVariableOptime_distributed_83958<time_distributed/batch_normalization/AssignMovingAvg/mul:z:0)^time_distributed/StatefulPartitionedCallD^time_distributed/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0�
<time_distributed/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Etime_distributed/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOptime_distributed_83960)^time_distributed/StatefulPartitionedCall*
_output_shapes	
:�*
dtype0�
:time_distributed/batch_normalization/AssignMovingAvg_1/subSubMtime_distributed/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0?time_distributed/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
:time_distributed/batch_normalization/AssignMovingAvg_1/mulMul>time_distributed/batch_normalization/AssignMovingAvg_1/sub:z:0Etime_distributed/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
6time_distributed/batch_normalization/AssignMovingAvg_1AssignSubVariableOptime_distributed_83960>time_distributed/batch_normalization/AssignMovingAvg_1/mul:z:0)^time_distributed/StatefulPartitionedCallF^time_distributed/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0y
4time_distributed/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
2time_distributed/batch_normalization/batchnorm/addAddV2?time_distributed/batch_normalization/moments/Squeeze_1:output:0=time_distributed/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
4time_distributed/batch_normalization/batchnorm/RsqrtRsqrt6time_distributed/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Atime_distributed/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOptime_distributed_83962*
_output_shapes	
:�*
dtype0�
2time_distributed/batch_normalization/batchnorm/mulMul8time_distributed/batch_normalization/batchnorm/Rsqrt:y:0Itime_distributed/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
4time_distributed/batch_normalization/batchnorm/mul_1Mul)time_distributed/dense/Relu:activations:06time_distributed/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
4time_distributed/batch_normalization/batchnorm/mul_2Mul=time_distributed/batch_normalization/moments/Squeeze:output:06time_distributed/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
=time_distributed/batch_normalization/batchnorm/ReadVariableOpReadVariableOptime_distributed_83964*
_output_shapes	
:�*
dtype0�
2time_distributed/batch_normalization/batchnorm/subSubEtime_distributed/batch_normalization/batchnorm/ReadVariableOp:value:08time_distributed/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
4time_distributed/batch_normalization/batchnorm/add_1AddV28time_distributed/batch_normalization/batchnorm/mul_1:z:06time_distributed/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������k
&time_distributed/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
$time_distributed/dropout/dropout/MulMul8time_distributed/batch_normalization/batchnorm/add_1:z:0/time_distributed/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
&time_distributed/dropout/dropout/ShapeShape8time_distributed/batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
=time_distributed/dropout/dropout/random_uniform/RandomUniformRandomUniform/time_distributed/dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*t
/time_distributed/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
-time_distributed/dropout/dropout/GreaterEqualGreaterEqualFtime_distributed/dropout/dropout/random_uniform/RandomUniform:output:08time_distributed/dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������m
(time_distributed/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
)time_distributed/dropout/dropout/SelectV2SelectV21time_distributed/dropout/dropout/GreaterEqual:z:0(time_distributed/dropout/dropout/Mul:z:01time_distributed/dropout/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
.time_distributed/dense_1/MatMul/ReadVariableOpReadVariableOptime_distributed_83966* 
_output_shapes
:
��*
dtype0�
time_distributed/dense_1/MatMulMatMul2time_distributed/dropout/dropout/SelectV2:output:06time_distributed/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/time_distributed/dense_1/BiasAdd/ReadVariableOpReadVariableOptime_distributed_83968*
_output_shapes	
:�*
dtype0�
 time_distributed/dense_1/BiasAddBiasAdd)time_distributed/dense_1/MatMul:product:07time_distributed/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
time_distributed/dense_1/ReluRelu)time_distributed/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Etime_distributed/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
3time_distributed/batch_normalization_1/moments/meanMean+time_distributed/dense_1/Relu:activations:0Ntime_distributed/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
;time_distributed/batch_normalization_1/moments/StopGradientStopGradient<time_distributed/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	��
@time_distributed/batch_normalization_1/moments/SquaredDifferenceSquaredDifference+time_distributed/dense_1/Relu:activations:0Dtime_distributed/batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
Itime_distributed/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
7time_distributed/batch_normalization_1/moments/varianceMeanDtime_distributed/batch_normalization_1/moments/SquaredDifference:z:0Rtime_distributed/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
6time_distributed/batch_normalization_1/moments/SqueezeSqueeze<time_distributed/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
8time_distributed/batch_normalization_1/moments/Squeeze_1Squeeze@time_distributed/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
<time_distributed/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Etime_distributed/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOptime_distributed_83970)^time_distributed/StatefulPartitionedCall*
_output_shapes	
:�*
dtype0�
:time_distributed/batch_normalization_1/AssignMovingAvg/subSubMtime_distributed/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0?time_distributed/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
:time_distributed/batch_normalization_1/AssignMovingAvg/mulMul>time_distributed/batch_normalization_1/AssignMovingAvg/sub:z:0Etime_distributed/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
6time_distributed/batch_normalization_1/AssignMovingAvgAssignSubVariableOptime_distributed_83970>time_distributed/batch_normalization_1/AssignMovingAvg/mul:z:0)^time_distributed/StatefulPartitionedCallF^time_distributed/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0�
>time_distributed/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Gtime_distributed/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOptime_distributed_83972)^time_distributed/StatefulPartitionedCall*
_output_shapes	
:�*
dtype0�
<time_distributed/batch_normalization_1/AssignMovingAvg_1/subSubOtime_distributed/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0Atime_distributed/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
<time_distributed/batch_normalization_1/AssignMovingAvg_1/mulMul@time_distributed/batch_normalization_1/AssignMovingAvg_1/sub:z:0Gtime_distributed/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
8time_distributed/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOptime_distributed_83972@time_distributed/batch_normalization_1/AssignMovingAvg_1/mul:z:0)^time_distributed/StatefulPartitionedCallH^time_distributed/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0{
6time_distributed/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4time_distributed/batch_normalization_1/batchnorm/addAddV2Atime_distributed/batch_normalization_1/moments/Squeeze_1:output:0?time_distributed/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
6time_distributed/batch_normalization_1/batchnorm/RsqrtRsqrt8time_distributed/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Ctime_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOptime_distributed_83974*
_output_shapes	
:�*
dtype0�
4time_distributed/batch_normalization_1/batchnorm/mulMul:time_distributed/batch_normalization_1/batchnorm/Rsqrt:y:0Ktime_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
6time_distributed/batch_normalization_1/batchnorm/mul_1Mul+time_distributed/dense_1/Relu:activations:08time_distributed/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
6time_distributed/batch_normalization_1/batchnorm/mul_2Mul?time_distributed/batch_normalization_1/moments/Squeeze:output:08time_distributed/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
?time_distributed/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOptime_distributed_83976*
_output_shapes	
:�*
dtype0�
4time_distributed/batch_normalization_1/batchnorm/subSubGtime_distributed/batch_normalization_1/batchnorm/ReadVariableOp:value:0:time_distributed/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
6time_distributed/batch_normalization_1/batchnorm/add_1AddV2:time_distributed/batch_normalization_1/batchnorm/mul_1:z:08time_distributed/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������m
(time_distributed/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
&time_distributed/dropout_1/dropout/MulMul:time_distributed/batch_normalization_1/batchnorm/add_1:z:01time_distributed/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
(time_distributed/dropout_1/dropout/ShapeShape:time_distributed/batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
?time_distributed/dropout_1/dropout/random_uniform/RandomUniformRandomUniform1time_distributed/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed2*

seed*v
1time_distributed/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
/time_distributed/dropout_1/dropout/GreaterEqualGreaterEqualHtime_distributed/dropout_1/dropout/random_uniform/RandomUniform:output:0:time_distributed/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������o
*time_distributed/dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
+time_distributed/dropout_1/dropout/SelectV2SelectV23time_distributed/dropout_1/dropout/GreaterEqual:z:0*time_distributed/dropout_1/dropout/Mul:z:03time_distributed/dropout_1/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
.time_distributed/dense_2/MatMul/ReadVariableOpReadVariableOptime_distributed_83978* 
_output_shapes
:
��*
dtype0�
time_distributed/dense_2/MatMulMatMul4time_distributed/dropout_1/dropout/SelectV2:output:06time_distributed/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/time_distributed/dense_2/BiasAdd/ReadVariableOpReadVariableOptime_distributed_83980*
_output_shapes	
:�*
dtype0�
 time_distributed/dense_2/BiasAddBiasAdd)time_distributed/dense_2/MatMul:product:07time_distributed/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
time_distributed/dense_2/ReluRelu)time_distributed/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Etime_distributed/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
3time_distributed/batch_normalization_2/moments/meanMean+time_distributed/dense_2/Relu:activations:0Ntime_distributed/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
;time_distributed/batch_normalization_2/moments/StopGradientStopGradient<time_distributed/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	��
@time_distributed/batch_normalization_2/moments/SquaredDifferenceSquaredDifference+time_distributed/dense_2/Relu:activations:0Dtime_distributed/batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
Itime_distributed/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
7time_distributed/batch_normalization_2/moments/varianceMeanDtime_distributed/batch_normalization_2/moments/SquaredDifference:z:0Rtime_distributed/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
6time_distributed/batch_normalization_2/moments/SqueezeSqueeze<time_distributed/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
8time_distributed/batch_normalization_2/moments/Squeeze_1Squeeze@time_distributed/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
<time_distributed/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Etime_distributed/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOptime_distributed_83982)^time_distributed/StatefulPartitionedCall*
_output_shapes	
:�*
dtype0�
:time_distributed/batch_normalization_2/AssignMovingAvg/subSubMtime_distributed/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0?time_distributed/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
:time_distributed/batch_normalization_2/AssignMovingAvg/mulMul>time_distributed/batch_normalization_2/AssignMovingAvg/sub:z:0Etime_distributed/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
6time_distributed/batch_normalization_2/AssignMovingAvgAssignSubVariableOptime_distributed_83982>time_distributed/batch_normalization_2/AssignMovingAvg/mul:z:0)^time_distributed/StatefulPartitionedCallF^time_distributed/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0�
>time_distributed/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Gtime_distributed/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOptime_distributed_83984)^time_distributed/StatefulPartitionedCall*
_output_shapes	
:�*
dtype0�
<time_distributed/batch_normalization_2/AssignMovingAvg_1/subSubOtime_distributed/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0Atime_distributed/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
<time_distributed/batch_normalization_2/AssignMovingAvg_1/mulMul@time_distributed/batch_normalization_2/AssignMovingAvg_1/sub:z:0Gtime_distributed/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
8time_distributed/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOptime_distributed_83984@time_distributed/batch_normalization_2/AssignMovingAvg_1/mul:z:0)^time_distributed/StatefulPartitionedCallH^time_distributed/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0{
6time_distributed/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4time_distributed/batch_normalization_2/batchnorm/addAddV2Atime_distributed/batch_normalization_2/moments/Squeeze_1:output:0?time_distributed/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
6time_distributed/batch_normalization_2/batchnorm/RsqrtRsqrt8time_distributed/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Ctime_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOptime_distributed_83986*
_output_shapes	
:�*
dtype0�
4time_distributed/batch_normalization_2/batchnorm/mulMul:time_distributed/batch_normalization_2/batchnorm/Rsqrt:y:0Ktime_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
6time_distributed/batch_normalization_2/batchnorm/mul_1Mul+time_distributed/dense_2/Relu:activations:08time_distributed/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
6time_distributed/batch_normalization_2/batchnorm/mul_2Mul?time_distributed/batch_normalization_2/moments/Squeeze:output:08time_distributed/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
?time_distributed/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOptime_distributed_83988*
_output_shapes	
:�*
dtype0�
4time_distributed/batch_normalization_2/batchnorm/subSubGtime_distributed/batch_normalization_2/batchnorm/ReadVariableOp:value:0:time_distributed/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
6time_distributed/batch_normalization_2/batchnorm/add_1AddV2:time_distributed/batch_normalization_2/batchnorm/mul_1:z:08time_distributed/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������m
(time_distributed/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
&time_distributed/dropout_2/dropout/MulMul:time_distributed/batch_normalization_2/batchnorm/add_1:z:01time_distributed/dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
(time_distributed/dropout_2/dropout/ShapeShape:time_distributed/batch_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
?time_distributed/dropout_2/dropout/random_uniform/RandomUniformRandomUniform1time_distributed/dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed2*

seed*v
1time_distributed/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
/time_distributed/dropout_2/dropout/GreaterEqualGreaterEqualHtime_distributed/dropout_2/dropout/random_uniform/RandomUniform:output:0:time_distributed/dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������o
*time_distributed/dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
+time_distributed/dropout_2/dropout/SelectV2SelectV23time_distributed/dropout_2/dropout/GreaterEqual:z:0*time_distributed/dropout_2/dropout/Mul:z:03time_distributed/dropout_2/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
.time_distributed/dense_3/MatMul/ReadVariableOpReadVariableOptime_distributed_83990* 
_output_shapes
:
��*
dtype0�
time_distributed/dense_3/MatMulMatMul4time_distributed/dropout_2/dropout/SelectV2:output:06time_distributed/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/time_distributed/dense_3/BiasAdd/ReadVariableOpReadVariableOptime_distributed_83992*
_output_shapes	
:�*
dtype0�
 time_distributed/dense_3/BiasAddBiasAdd)time_distributed/dense_3/MatMul:product:07time_distributed/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
time_distributed/dense_3/ReluRelu)time_distributed/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
&self_attention/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:01time_distributed/StatefulPartitionedCall:output:01time_distributed/StatefulPartitionedCall:output:0self_attention_84248self_attention_84250self_attention_84252*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������#�*%
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *R
fMRK
I__inference_self_attention_layer_call_and_return_conditional_losses_84247�
"att_layer2/StatefulPartitionedCallStatefulPartitionedCall/self_attention/StatefulPartitionedCall:output:0att_layer2_84317att_layer2_84319att_layer2_84321*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *N
fIRG
E__inference_att_layer2_layer_call_and_return_conditional_losses_84316{
IdentityIdentity+att_layer2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp#^att_layer2/StatefulPartitionedCall'^self_attention/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall5^time_distributed/batch_normalization/AssignMovingAvgD^time_distributed/batch_normalization/AssignMovingAvg/ReadVariableOp7^time_distributed/batch_normalization/AssignMovingAvg_1F^time_distributed/batch_normalization/AssignMovingAvg_1/ReadVariableOp>^time_distributed/batch_normalization/batchnorm/ReadVariableOpB^time_distributed/batch_normalization/batchnorm/mul/ReadVariableOp7^time_distributed/batch_normalization_1/AssignMovingAvgF^time_distributed/batch_normalization_1/AssignMovingAvg/ReadVariableOp9^time_distributed/batch_normalization_1/AssignMovingAvg_1H^time_distributed/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp@^time_distributed/batch_normalization_1/batchnorm/ReadVariableOpD^time_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOp7^time_distributed/batch_normalization_2/AssignMovingAvgF^time_distributed/batch_normalization_2/AssignMovingAvg/ReadVariableOp9^time_distributed/batch_normalization_2/AssignMovingAvg_1H^time_distributed/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp@^time_distributed/batch_normalization_2/batchnorm/ReadVariableOpD^time_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOp.^time_distributed/dense/BiasAdd/ReadVariableOp-^time_distributed/dense/MatMul/ReadVariableOp0^time_distributed/dense_1/BiasAdd/ReadVariableOp/^time_distributed/dense_1/MatMul/ReadVariableOp0^time_distributed/dense_2/BiasAdd/ReadVariableOp/^time_distributed/dense_2/MatMul/ReadVariableOp0^time_distributed/dense_3/BiasAdd/ReadVariableOp/^time_distributed/dense_3/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:���������#�: : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"att_layer2/StatefulPartitionedCall"att_layer2/StatefulPartitionedCall2P
&self_attention/StatefulPartitionedCall&self_attention/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2�
Ctime_distributed/batch_normalization/AssignMovingAvg/ReadVariableOpCtime_distributed/batch_normalization/AssignMovingAvg/ReadVariableOp2�
Etime_distributed/batch_normalization/AssignMovingAvg_1/ReadVariableOpEtime_distributed/batch_normalization/AssignMovingAvg_1/ReadVariableOp2p
6time_distributed/batch_normalization/AssignMovingAvg_16time_distributed/batch_normalization/AssignMovingAvg_12l
4time_distributed/batch_normalization/AssignMovingAvg4time_distributed/batch_normalization/AssignMovingAvg2~
=time_distributed/batch_normalization/batchnorm/ReadVariableOp=time_distributed/batch_normalization/batchnorm/ReadVariableOp2�
Atime_distributed/batch_normalization/batchnorm/mul/ReadVariableOpAtime_distributed/batch_normalization/batchnorm/mul/ReadVariableOp2�
Etime_distributed/batch_normalization_1/AssignMovingAvg/ReadVariableOpEtime_distributed/batch_normalization_1/AssignMovingAvg/ReadVariableOp2�
Gtime_distributed/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpGtime_distributed/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2t
8time_distributed/batch_normalization_1/AssignMovingAvg_18time_distributed/batch_normalization_1/AssignMovingAvg_12p
6time_distributed/batch_normalization_1/AssignMovingAvg6time_distributed/batch_normalization_1/AssignMovingAvg2�
?time_distributed/batch_normalization_1/batchnorm/ReadVariableOp?time_distributed/batch_normalization_1/batchnorm/ReadVariableOp2�
Ctime_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOpCtime_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOp2�
Etime_distributed/batch_normalization_2/AssignMovingAvg/ReadVariableOpEtime_distributed/batch_normalization_2/AssignMovingAvg/ReadVariableOp2�
Gtime_distributed/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpGtime_distributed/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2t
8time_distributed/batch_normalization_2/AssignMovingAvg_18time_distributed/batch_normalization_2/AssignMovingAvg_12p
6time_distributed/batch_normalization_2/AssignMovingAvg6time_distributed/batch_normalization_2/AssignMovingAvg2�
?time_distributed/batch_normalization_2/batchnorm/ReadVariableOp?time_distributed/batch_normalization_2/batchnorm/ReadVariableOp2�
Ctime_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOpCtime_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOp2^
-time_distributed/dense/BiasAdd/ReadVariableOp-time_distributed/dense/BiasAdd/ReadVariableOp2\
,time_distributed/dense/MatMul/ReadVariableOp,time_distributed/dense/MatMul/ReadVariableOp2b
/time_distributed/dense_1/BiasAdd/ReadVariableOp/time_distributed/dense_1/BiasAdd/ReadVariableOp2`
.time_distributed/dense_1/MatMul/ReadVariableOp.time_distributed/dense_1/MatMul/ReadVariableOp2b
/time_distributed/dense_2/BiasAdd/ReadVariableOp/time_distributed/dense_2/BiasAdd/ReadVariableOp2`
.time_distributed/dense_2/MatMul/ReadVariableOp.time_distributed/dense_2/MatMul/ReadVariableOp2b
/time_distributed/dense_3/BiasAdd/ReadVariableOp/time_distributed/dense_3/BiasAdd/ReadVariableOp2`
.time_distributed/dense_3/MatMul/ReadVariableOp.time_distributed/dense_3/MatMul/ReadVariableOp:%!

_user_specified_name84321:%!

_user_specified_name84319:%!

_user_specified_name84317:%!

_user_specified_name84252:%!

_user_specified_name84250:%!

_user_specified_name84248:%!

_user_specified_name83992:%!

_user_specified_name83990:%!

_user_specified_name83988:%!

_user_specified_name83986:%!

_user_specified_name83984:%!

_user_specified_name83982:%!

_user_specified_name83980:%!

_user_specified_name83978:%!

_user_specified_name83976:%!

_user_specified_name83974:%
!

_user_specified_name83972:%	!

_user_specified_name83970:%!

_user_specified_name83968:%!

_user_specified_name83966:%!

_user_specified_name83964:%!

_user_specified_name83962:%!

_user_specified_name83960:%!

_user_specified_name83958:%!

_user_specified_name83956:%!

_user_specified_name83954:U Q
,
_output_shapes
:���������#�
!
_user_specified_name	input_5
�

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_86451

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
K__inference_time_distributed_layer_call_and_return_conditional_losses_85914

inputsE
1news_encoder_dense_matmul_readvariableop_resource:
��A
2news_encoder_dense_biasadd_readvariableop_resource:	�Q
Bnews_encoder_batch_normalization_batchnorm_readvariableop_resource:	�U
Fnews_encoder_batch_normalization_batchnorm_mul_readvariableop_resource:	�S
Dnews_encoder_batch_normalization_batchnorm_readvariableop_1_resource:	�S
Dnews_encoder_batch_normalization_batchnorm_readvariableop_2_resource:	�G
3news_encoder_dense_1_matmul_readvariableop_resource:
��C
4news_encoder_dense_1_biasadd_readvariableop_resource:	�S
Dnews_encoder_batch_normalization_1_batchnorm_readvariableop_resource:	�W
Hnews_encoder_batch_normalization_1_batchnorm_mul_readvariableop_resource:	�U
Fnews_encoder_batch_normalization_1_batchnorm_readvariableop_1_resource:	�U
Fnews_encoder_batch_normalization_1_batchnorm_readvariableop_2_resource:	�G
3news_encoder_dense_2_matmul_readvariableop_resource:
��C
4news_encoder_dense_2_biasadd_readvariableop_resource:	�S
Dnews_encoder_batch_normalization_2_batchnorm_readvariableop_resource:	�W
Hnews_encoder_batch_normalization_2_batchnorm_mul_readvariableop_resource:	�U
Fnews_encoder_batch_normalization_2_batchnorm_readvariableop_1_resource:	�U
Fnews_encoder_batch_normalization_2_batchnorm_readvariableop_2_resource:	�G
3news_encoder_dense_3_matmul_readvariableop_resource:
��C
4news_encoder_dense_3_biasadd_readvariableop_resource:	�
identity��9news_encoder/batch_normalization/batchnorm/ReadVariableOp�;news_encoder/batch_normalization/batchnorm/ReadVariableOp_1�;news_encoder/batch_normalization/batchnorm/ReadVariableOp_2�=news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp�;news_encoder/batch_normalization_1/batchnorm/ReadVariableOp�=news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1�=news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2�?news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp�;news_encoder/batch_normalization_2/batchnorm/ReadVariableOp�=news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1�=news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2�?news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp�)news_encoder/dense/BiasAdd/ReadVariableOp�(news_encoder/dense/MatMul/ReadVariableOp�+news_encoder/dense_1/BiasAdd/ReadVariableOp�*news_encoder/dense_1/MatMul/ReadVariableOp�+news_encoder/dense_2/BiasAdd/ReadVariableOp�*news_encoder/dense_2/MatMul/ReadVariableOp�+news_encoder/dense_3/BiasAdd/ReadVariableOp�*news_encoder/dense_3/MatMul/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
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
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:�����������
(news_encoder/dense/MatMul/ReadVariableOpReadVariableOp1news_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
news_encoder/dense/MatMulMatMulReshape:output:00news_encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
)news_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp2news_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
news_encoder/dense/BiasAddBiasAdd#news_encoder/dense/MatMul:product:01news_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������w
news_encoder/dense/ReluRelu#news_encoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
9news_encoder/batch_normalization/batchnorm/ReadVariableOpReadVariableOpBnews_encoder_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0u
0news_encoder/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
.news_encoder/batch_normalization/batchnorm/addAddV2Anews_encoder/batch_normalization/batchnorm/ReadVariableOp:value:09news_encoder/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
0news_encoder/batch_normalization/batchnorm/RsqrtRsqrt2news_encoder/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
=news_encoder/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpFnews_encoder_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.news_encoder/batch_normalization/batchnorm/mulMul4news_encoder/batch_normalization/batchnorm/Rsqrt:y:0Enews_encoder/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
0news_encoder/batch_normalization/batchnorm/mul_1Mul%news_encoder/dense/Relu:activations:02news_encoder/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
;news_encoder/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpDnews_encoder_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization/batchnorm/mul_2MulCnews_encoder/batch_normalization/batchnorm/ReadVariableOp_1:value:02news_encoder/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
;news_encoder/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpDnews_encoder_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
.news_encoder/batch_normalization/batchnorm/subSubCnews_encoder/batch_normalization/batchnorm/ReadVariableOp_2:value:04news_encoder/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
0news_encoder/batch_normalization/batchnorm/add_1AddV24news_encoder/batch_normalization/batchnorm/mul_1:z:02news_encoder/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
news_encoder/dropout/IdentityIdentity4news_encoder/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
*news_encoder/dense_1/MatMul/ReadVariableOpReadVariableOp3news_encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
news_encoder/dense_1/MatMulMatMul&news_encoder/dropout/Identity:output:02news_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+news_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp4news_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
news_encoder/dense_1/BiasAddBiasAdd%news_encoder/dense_1/MatMul:product:03news_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
news_encoder/dense_1/ReluRelu%news_encoder/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
;news_encoder/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpDnews_encoder_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0w
2news_encoder/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
0news_encoder/batch_normalization_1/batchnorm/addAddV2Cnews_encoder/batch_normalization_1/batchnorm/ReadVariableOp:value:0;news_encoder/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_1/batchnorm/RsqrtRsqrt4news_encoder/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
?news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpHnews_encoder_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization_1/batchnorm/mulMul6news_encoder/batch_normalization_1/batchnorm/Rsqrt:y:0Gnews_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_1/batchnorm/mul_1Mul'news_encoder/dense_1/Relu:activations:04news_encoder/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
=news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpFnews_encoder_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
2news_encoder/batch_normalization_1/batchnorm/mul_2MulEnews_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1:value:04news_encoder/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
=news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpFnews_encoder_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization_1/batchnorm/subSubEnews_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2:value:06news_encoder/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_1/batchnorm/add_1AddV26news_encoder/batch_normalization_1/batchnorm/mul_1:z:04news_encoder/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
news_encoder/dropout_1/IdentityIdentity6news_encoder/batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
*news_encoder/dense_2/MatMul/ReadVariableOpReadVariableOp3news_encoder_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
news_encoder/dense_2/MatMulMatMul(news_encoder/dropout_1/Identity:output:02news_encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+news_encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp4news_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
news_encoder/dense_2/BiasAddBiasAdd%news_encoder/dense_2/MatMul:product:03news_encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
news_encoder/dense_2/ReluRelu%news_encoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
;news_encoder/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpDnews_encoder_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0w
2news_encoder/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
0news_encoder/batch_normalization_2/batchnorm/addAddV2Cnews_encoder/batch_normalization_2/batchnorm/ReadVariableOp:value:0;news_encoder/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_2/batchnorm/RsqrtRsqrt4news_encoder/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
?news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpHnews_encoder_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization_2/batchnorm/mulMul6news_encoder/batch_normalization_2/batchnorm/Rsqrt:y:0Gnews_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_2/batchnorm/mul_1Mul'news_encoder/dense_2/Relu:activations:04news_encoder/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
=news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpFnews_encoder_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
2news_encoder/batch_normalization_2/batchnorm/mul_2MulEnews_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1:value:04news_encoder/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
=news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpFnews_encoder_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization_2/batchnorm/subSubEnews_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2:value:06news_encoder/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_2/batchnorm/add_1AddV26news_encoder/batch_normalization_2/batchnorm/mul_1:z:04news_encoder/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
news_encoder/dropout_2/IdentityIdentity6news_encoder/batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
*news_encoder/dense_3/MatMul/ReadVariableOpReadVariableOp3news_encoder_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
news_encoder/dense_3/MatMulMatMul(news_encoder/dropout_2/Identity:output:02news_encoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+news_encoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp4news_encoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
news_encoder/dense_3/BiasAddBiasAdd%news_encoder/dense_3/MatMul:product:03news_encoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
news_encoder/dense_3/ReluRelu%news_encoder/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape'news_encoder/dense_3/Relu:activations:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:��������������������	
NoOpNoOp:^news_encoder/batch_normalization/batchnorm/ReadVariableOp<^news_encoder/batch_normalization/batchnorm/ReadVariableOp_1<^news_encoder/batch_normalization/batchnorm/ReadVariableOp_2>^news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp<^news_encoder/batch_normalization_1/batchnorm/ReadVariableOp>^news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1>^news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2@^news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp<^news_encoder/batch_normalization_2/batchnorm/ReadVariableOp>^news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1>^news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2@^news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp*^news_encoder/dense/BiasAdd/ReadVariableOp)^news_encoder/dense/MatMul/ReadVariableOp,^news_encoder/dense_1/BiasAdd/ReadVariableOp+^news_encoder/dense_1/MatMul/ReadVariableOp,^news_encoder/dense_2/BiasAdd/ReadVariableOp+^news_encoder/dense_2/MatMul/ReadVariableOp,^news_encoder/dense_3/BiasAdd/ReadVariableOp+^news_encoder/dense_3/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�������������������: : : : : : : : : : : : : : : : : : : : 2z
;news_encoder/batch_normalization/batchnorm/ReadVariableOp_1;news_encoder/batch_normalization/batchnorm/ReadVariableOp_12z
;news_encoder/batch_normalization/batchnorm/ReadVariableOp_2;news_encoder/batch_normalization/batchnorm/ReadVariableOp_22v
9news_encoder/batch_normalization/batchnorm/ReadVariableOp9news_encoder/batch_normalization/batchnorm/ReadVariableOp2~
=news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp=news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp2~
=news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1=news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_12~
=news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2=news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_22z
;news_encoder/batch_normalization_1/batchnorm/ReadVariableOp;news_encoder/batch_normalization_1/batchnorm/ReadVariableOp2�
?news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp?news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp2~
=news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1=news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_12~
=news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2=news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_22z
;news_encoder/batch_normalization_2/batchnorm/ReadVariableOp;news_encoder/batch_normalization_2/batchnorm/ReadVariableOp2�
?news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp?news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp2V
)news_encoder/dense/BiasAdd/ReadVariableOp)news_encoder/dense/BiasAdd/ReadVariableOp2T
(news_encoder/dense/MatMul/ReadVariableOp(news_encoder/dense/MatMul/ReadVariableOp2Z
+news_encoder/dense_1/BiasAdd/ReadVariableOp+news_encoder/dense_1/BiasAdd/ReadVariableOp2X
*news_encoder/dense_1/MatMul/ReadVariableOp*news_encoder/dense_1/MatMul/ReadVariableOp2Z
+news_encoder/dense_2/BiasAdd/ReadVariableOp+news_encoder/dense_2/BiasAdd/ReadVariableOp2X
*news_encoder/dense_2/MatMul/ReadVariableOp*news_encoder/dense_2/MatMul/ReadVariableOp2Z
+news_encoder/dense_3/BiasAdd/ReadVariableOp+news_encoder/dense_3/BiasAdd/ReadVariableOp2X
*news_encoder/dense_3/MatMul/ReadVariableOp*news_encoder/dense_3/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
2__inference_time_distributed_1_layer_call_fn_85296

inputs
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
��

unknown_18:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*6
_read_only_resource_inputs
	
*<
config_proto,*

CPU

GPU 2J 8R(���������� *V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_83659}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�������������������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name85292:%!

_user_specified_name85290:%!

_user_specified_name85288:%!

_user_specified_name85286:%!

_user_specified_name85284:%!

_user_specified_name85282:%!

_user_specified_name85280:%!

_user_specified_name85278:%!

_user_specified_name85276:%!

_user_specified_name85274:%
!

_user_specified_name85272:%	!

_user_specified_name85270:%!

_user_specified_name85268:%!

_user_specified_name85266:%!

_user_specified_name85264:%!

_user_specified_name85262:%!

_user_specified_name85260:%!

_user_specified_name85258:%!

_user_specified_name85256:%!

_user_specified_name85254:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
'__inference_dense_2_layer_call_fn_86465

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_83277p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name86461:%!

_user_specified_name86459:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
3__inference_batch_normalization_layer_call_fn_86248

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_83002p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name86244:%!

_user_specified_name86242:%!

_user_specified_name86240:%!

_user_specified_name86238:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_83002

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_dense_1_layer_call_and_return_conditional_losses_83239

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_activation_layer_call_fn_85568

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_84873i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:������������������:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�/
�
E__inference_att_layer2_layer_call_and_return_conditional_losses_84316

inputs3
shape_1_readvariableop_resource:
��*
add_readvariableop_resource:	�2
shape_3_readvariableop_resource:	�
identity��add/ReadVariableOp�transpose/ReadVariableOp�transpose_1/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"�  �   S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������z
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       |
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0* 
_output_shapes
:
��`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"�  ����h
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
��i
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*(
_output_shapes
:����������S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#T
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:w
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*,
_output_shapes
:���������#�k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0s
addAddV2Reshape_2:output:0add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������#�L
TanhTanhadd:z:0*
T0*,
_output_shapes
:���������#�M
Shape_2ShapeTanh:y:0*
T0*
_output_shapes
::��U
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numw
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"�      S
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   k
	Reshape_3ReshapeTanh:y:0Reshape_3/shape:output:0*
T0*(
_output_shapes
:����������{
transpose_1/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes
:	�*
dtype0a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       �
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes
:	�`
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"�   ����i
	Reshape_4Reshapetranspose_1:y:0Reshape_4/shape:output:0*
T0*
_output_shapes
:	�l
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:���������S
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#S
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:x
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:���������#o
SqueezeSqueezeReshape_5:output:0*
T0*'
_output_shapes
:���������#*
squeeze_dims
N
ExpExpSqueeze:output:0*
T0*'
_output_shapes
:���������#`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������v
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3`
add_1AddV2Sum:output:0add_1/y:output:0*
T0*'
_output_shapes
:���������X
truedivRealDivExp:y:0	add_1:z:0*
T0*'
_output_shapes
:���������#Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������t

ExpandDims
ExpandDimstruediv:z:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������#^
mulMulinputsExpandDims:output:0*
T0*,
_output_shapes
:���������#�Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :j
Sum_1Summul:z:0 Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:����������^
IdentityIdentitySum_1:output:0^NoOp*
T0*(
_output_shapes
:����������o
NoOpNoOp^add/ReadVariableOp^transpose/ReadVariableOp^transpose_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:���������#�: : : 2(
add/ReadVariableOpadd/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
,
_output_shapes
:���������#�
 
_user_specified_nameinputs
�
�
0__inference_time_distributed_layer_call_fn_85618

inputs
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
��

unknown_18:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*0
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_83805}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�������������������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name85614:%!

_user_specified_name85612:%!

_user_specified_name85610:%!

_user_specified_name85608:%!

_user_specified_name85606:%!

_user_specified_name85604:%!

_user_specified_name85602:%!

_user_specified_name85600:%!

_user_specified_name85598:%!

_user_specified_name85596:%
!

_user_specified_name85594:%	!

_user_specified_name85592:%!

_user_specified_name85590:%!

_user_specified_name85588:%!

_user_specified_name85586:%!

_user_specified_name85584:%!

_user_specified_name85582:%!

_user_specified_name85580:%!

_user_specified_name85578:%!

_user_specified_name85576:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
��
�E
__inference__traced_save_87070
file_prefix7
#read_disablecopyonread_dense_kernel:
��2
#read_1_disablecopyonread_dense_bias:	�A
2read_2_disablecopyonread_batch_normalization_gamma:	�@
1read_3_disablecopyonread_batch_normalization_beta:	�;
'read_4_disablecopyonread_dense_1_kernel:
��4
%read_5_disablecopyonread_dense_1_bias:	�C
4read_6_disablecopyonread_batch_normalization_1_gamma:	�B
3read_7_disablecopyonread_batch_normalization_1_beta:	�;
'read_8_disablecopyonread_dense_2_kernel:
��4
%read_9_disablecopyonread_dense_2_bias:	�D
5read_10_disablecopyonread_batch_normalization_2_gamma:	�C
4read_11_disablecopyonread_batch_normalization_2_beta:	�<
(read_12_disablecopyonread_dense_3_kernel:
��5
&read_13_disablecopyonread_dense_3_bias:	�H
9read_14_disablecopyonread_batch_normalization_moving_mean:	�L
=read_15_disablecopyonread_batch_normalization_moving_variance:	�J
;read_16_disablecopyonread_batch_normalization_1_moving_mean:	�N
?read_17_disablecopyonread_batch_normalization_1_moving_variance:	�J
;read_18_disablecopyonread_batch_normalization_2_moving_mean:	�N
?read_19_disablecopyonread_batch_normalization_2_moving_variance:	�?
+read_20_disablecopyonread_self_attention_wq:
��?
+read_21_disablecopyonread_self_attention_wk:
��?
+read_22_disablecopyonread_self_attention_wv:
��:
&read_23_disablecopyonread_att_layer2_w:
��5
&read_24_disablecopyonread_att_layer2_b:	�9
&read_25_disablecopyonread_att_layer2_q:	�-
#read_26_disablecopyonread_iteration:	 1
'read_27_disablecopyonread_learning_rate: A
-read_28_disablecopyonread_adam_m_dense_kernel:
��A
-read_29_disablecopyonread_adam_v_dense_kernel:
��:
+read_30_disablecopyonread_adam_m_dense_bias:	�:
+read_31_disablecopyonread_adam_v_dense_bias:	�I
:read_32_disablecopyonread_adam_m_batch_normalization_gamma:	�I
:read_33_disablecopyonread_adam_v_batch_normalization_gamma:	�H
9read_34_disablecopyonread_adam_m_batch_normalization_beta:	�H
9read_35_disablecopyonread_adam_v_batch_normalization_beta:	�C
/read_36_disablecopyonread_adam_m_dense_1_kernel:
��C
/read_37_disablecopyonread_adam_v_dense_1_kernel:
��<
-read_38_disablecopyonread_adam_m_dense_1_bias:	�<
-read_39_disablecopyonread_adam_v_dense_1_bias:	�K
<read_40_disablecopyonread_adam_m_batch_normalization_1_gamma:	�K
<read_41_disablecopyonread_adam_v_batch_normalization_1_gamma:	�J
;read_42_disablecopyonread_adam_m_batch_normalization_1_beta:	�J
;read_43_disablecopyonread_adam_v_batch_normalization_1_beta:	�C
/read_44_disablecopyonread_adam_m_dense_2_kernel:
��C
/read_45_disablecopyonread_adam_v_dense_2_kernel:
��<
-read_46_disablecopyonread_adam_m_dense_2_bias:	�<
-read_47_disablecopyonread_adam_v_dense_2_bias:	�K
<read_48_disablecopyonread_adam_m_batch_normalization_2_gamma:	�K
<read_49_disablecopyonread_adam_v_batch_normalization_2_gamma:	�J
;read_50_disablecopyonread_adam_m_batch_normalization_2_beta:	�J
;read_51_disablecopyonread_adam_v_batch_normalization_2_beta:	�C
/read_52_disablecopyonread_adam_m_dense_3_kernel:
��C
/read_53_disablecopyonread_adam_v_dense_3_kernel:
��<
-read_54_disablecopyonread_adam_m_dense_3_bias:	�<
-read_55_disablecopyonread_adam_v_dense_3_bias:	�F
2read_56_disablecopyonread_adam_m_self_attention_wq:
��F
2read_57_disablecopyonread_adam_v_self_attention_wq:
��F
2read_58_disablecopyonread_adam_m_self_attention_wk:
��F
2read_59_disablecopyonread_adam_v_self_attention_wk:
��F
2read_60_disablecopyonread_adam_m_self_attention_wv:
��F
2read_61_disablecopyonread_adam_v_self_attention_wv:
��A
-read_62_disablecopyonread_adam_m_att_layer2_w:
��A
-read_63_disablecopyonread_adam_v_att_layer2_w:
��<
-read_64_disablecopyonread_adam_m_att_layer2_b:	�<
-read_65_disablecopyonread_adam_v_att_layer2_b:	�@
-read_66_disablecopyonread_adam_m_att_layer2_q:	�@
-read_67_disablecopyonread_adam_v_att_layer2_q:	�)
read_68_disablecopyonread_total: )
read_69_disablecopyonread_count: 7
(read_70_disablecopyonread_true_positives:	�7
(read_71_disablecopyonread_true_negatives:	�8
)read_72_disablecopyonread_false_positives:	�8
)read_73_disablecopyonread_false_negatives:	�
savev2_const
identity_149��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��c

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_2/DisableCopyOnReadDisableCopyOnRead2read_2_disablecopyonread_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp2read_2_disablecopyonread_batch_normalization_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_3/DisableCopyOnReadDisableCopyOnRead1read_3_disablecopyonread_batch_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp1read_3_disablecopyonread_batch_normalization_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_dense_1_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��y
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_dense_1_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_6/DisableCopyOnReadDisableCopyOnRead4read_6_disablecopyonread_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp4read_6_disablecopyonread_batch_normalization_1_gamma^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_7/DisableCopyOnReadDisableCopyOnRead3read_7_disablecopyonread_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp3read_7_disablecopyonread_batch_normalization_1_beta^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_8/DisableCopyOnReadDisableCopyOnRead'read_8_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp'read_8_disablecopyonread_dense_2_kernel^Read_8/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��y
Read_9/DisableCopyOnReadDisableCopyOnRead%read_9_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp%read_9_disablecopyonread_dense_2_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_10/DisableCopyOnReadDisableCopyOnRead5read_10_disablecopyonread_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp5read_10_disablecopyonread_batch_normalization_2_gamma^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_11/DisableCopyOnReadDisableCopyOnRead4read_11_disablecopyonread_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp4read_11_disablecopyonread_batch_normalization_2_beta^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_12/DisableCopyOnReadDisableCopyOnRead(read_12_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp(read_12_disablecopyonread_dense_3_kernel^Read_12/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_13/DisableCopyOnReadDisableCopyOnRead&read_13_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp&read_13_disablecopyonread_dense_3_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_14/DisableCopyOnReadDisableCopyOnRead9read_14_disablecopyonread_batch_normalization_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp9read_14_disablecopyonread_batch_normalization_moving_mean^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_15/DisableCopyOnReadDisableCopyOnRead=read_15_disablecopyonread_batch_normalization_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp=read_15_disablecopyonread_batch_normalization_moving_variance^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_16/DisableCopyOnReadDisableCopyOnRead;read_16_disablecopyonread_batch_normalization_1_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp;read_16_disablecopyonread_batch_normalization_1_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_17/DisableCopyOnReadDisableCopyOnRead?read_17_disablecopyonread_batch_normalization_1_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp?read_17_disablecopyonread_batch_normalization_1_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_18/DisableCopyOnReadDisableCopyOnRead;read_18_disablecopyonread_batch_normalization_2_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp;read_18_disablecopyonread_batch_normalization_2_moving_mean^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_19/DisableCopyOnReadDisableCopyOnRead?read_19_disablecopyonread_batch_normalization_2_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp?read_19_disablecopyonread_batch_normalization_2_moving_variance^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_20/DisableCopyOnReadDisableCopyOnRead+read_20_disablecopyonread_self_attention_wq"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp+read_20_disablecopyonread_self_attention_wq^Read_20/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_21/DisableCopyOnReadDisableCopyOnRead+read_21_disablecopyonread_self_attention_wk"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp+read_21_disablecopyonread_self_attention_wk^Read_21/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_22/DisableCopyOnReadDisableCopyOnRead+read_22_disablecopyonread_self_attention_wv"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp+read_22_disablecopyonread_self_attention_wv^Read_22/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_23/DisableCopyOnReadDisableCopyOnRead&read_23_disablecopyonread_att_layer2_w"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp&read_23_disablecopyonread_att_layer2_w^Read_23/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_24/DisableCopyOnReadDisableCopyOnRead&read_24_disablecopyonread_att_layer2_b"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp&read_24_disablecopyonread_att_layer2_b^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_25/DisableCopyOnReadDisableCopyOnRead&read_25_disablecopyonread_att_layer2_q"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp&read_25_disablecopyonread_att_layer2_q^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:	�x
Read_26/DisableCopyOnReadDisableCopyOnRead#read_26_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp#read_26_disablecopyonread_iteration^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_27/DisableCopyOnReadDisableCopyOnRead'read_27_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp'read_27_disablecopyonread_learning_rate^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_28/DisableCopyOnReadDisableCopyOnRead-read_28_disablecopyonread_adam_m_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp-read_28_disablecopyonread_adam_m_dense_kernel^Read_28/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_29/DisableCopyOnReadDisableCopyOnRead-read_29_disablecopyonread_adam_v_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp-read_29_disablecopyonread_adam_v_dense_kernel^Read_29/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_30/DisableCopyOnReadDisableCopyOnRead+read_30_disablecopyonread_adam_m_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp+read_30_disablecopyonread_adam_m_dense_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_31/DisableCopyOnReadDisableCopyOnRead+read_31_disablecopyonread_adam_v_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp+read_31_disablecopyonread_adam_v_dense_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_32/DisableCopyOnReadDisableCopyOnRead:read_32_disablecopyonread_adam_m_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp:read_32_disablecopyonread_adam_m_batch_normalization_gamma^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_33/DisableCopyOnReadDisableCopyOnRead:read_33_disablecopyonread_adam_v_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp:read_33_disablecopyonread_adam_v_batch_normalization_gamma^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_34/DisableCopyOnReadDisableCopyOnRead9read_34_disablecopyonread_adam_m_batch_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp9read_34_disablecopyonread_adam_m_batch_normalization_beta^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_35/DisableCopyOnReadDisableCopyOnRead9read_35_disablecopyonread_adam_v_batch_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp9read_35_disablecopyonread_adam_v_batch_normalization_beta^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_36/DisableCopyOnReadDisableCopyOnRead/read_36_disablecopyonread_adam_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp/read_36_disablecopyonread_adam_m_dense_1_kernel^Read_36/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_37/DisableCopyOnReadDisableCopyOnRead/read_37_disablecopyonread_adam_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp/read_37_disablecopyonread_adam_v_dense_1_kernel^Read_37/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_38/DisableCopyOnReadDisableCopyOnRead-read_38_disablecopyonread_adam_m_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp-read_38_disablecopyonread_adam_m_dense_1_bias^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_39/DisableCopyOnReadDisableCopyOnRead-read_39_disablecopyonread_adam_v_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp-read_39_disablecopyonread_adam_v_dense_1_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_40/DisableCopyOnReadDisableCopyOnRead<read_40_disablecopyonread_adam_m_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp<read_40_disablecopyonread_adam_m_batch_normalization_1_gamma^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_41/DisableCopyOnReadDisableCopyOnRead<read_41_disablecopyonread_adam_v_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp<read_41_disablecopyonread_adam_v_batch_normalization_1_gamma^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_42/DisableCopyOnReadDisableCopyOnRead;read_42_disablecopyonread_adam_m_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp;read_42_disablecopyonread_adam_m_batch_normalization_1_beta^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_43/DisableCopyOnReadDisableCopyOnRead;read_43_disablecopyonread_adam_v_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp;read_43_disablecopyonread_adam_v_batch_normalization_1_beta^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_44/DisableCopyOnReadDisableCopyOnRead/read_44_disablecopyonread_adam_m_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp/read_44_disablecopyonread_adam_m_dense_2_kernel^Read_44/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_45/DisableCopyOnReadDisableCopyOnRead/read_45_disablecopyonread_adam_v_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp/read_45_disablecopyonread_adam_v_dense_2_kernel^Read_45/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_46/DisableCopyOnReadDisableCopyOnRead-read_46_disablecopyonread_adam_m_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp-read_46_disablecopyonread_adam_m_dense_2_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_47/DisableCopyOnReadDisableCopyOnRead-read_47_disablecopyonread_adam_v_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp-read_47_disablecopyonread_adam_v_dense_2_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_48/DisableCopyOnReadDisableCopyOnRead<read_48_disablecopyonread_adam_m_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp<read_48_disablecopyonread_adam_m_batch_normalization_2_gamma^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_49/DisableCopyOnReadDisableCopyOnRead<read_49_disablecopyonread_adam_v_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp<read_49_disablecopyonread_adam_v_batch_normalization_2_gamma^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_50/DisableCopyOnReadDisableCopyOnRead;read_50_disablecopyonread_adam_m_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp;read_50_disablecopyonread_adam_m_batch_normalization_2_beta^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_51/DisableCopyOnReadDisableCopyOnRead;read_51_disablecopyonread_adam_v_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp;read_51_disablecopyonread_adam_v_batch_normalization_2_beta^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_52/DisableCopyOnReadDisableCopyOnRead/read_52_disablecopyonread_adam_m_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp/read_52_disablecopyonread_adam_m_dense_3_kernel^Read_52/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_53/DisableCopyOnReadDisableCopyOnRead/read_53_disablecopyonread_adam_v_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp/read_53_disablecopyonread_adam_v_dense_3_kernel^Read_53/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_54/DisableCopyOnReadDisableCopyOnRead-read_54_disablecopyonread_adam_m_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp-read_54_disablecopyonread_adam_m_dense_3_bias^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_55/DisableCopyOnReadDisableCopyOnRead-read_55_disablecopyonread_adam_v_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp-read_55_disablecopyonread_adam_v_dense_3_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_56/DisableCopyOnReadDisableCopyOnRead2read_56_disablecopyonread_adam_m_self_attention_wq"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp2read_56_disablecopyonread_adam_m_self_attention_wq^Read_56/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_57/DisableCopyOnReadDisableCopyOnRead2read_57_disablecopyonread_adam_v_self_attention_wq"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp2read_57_disablecopyonread_adam_v_self_attention_wq^Read_57/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_58/DisableCopyOnReadDisableCopyOnRead2read_58_disablecopyonread_adam_m_self_attention_wk"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp2read_58_disablecopyonread_adam_m_self_attention_wk^Read_58/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_59/DisableCopyOnReadDisableCopyOnRead2read_59_disablecopyonread_adam_v_self_attention_wk"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp2read_59_disablecopyonread_adam_v_self_attention_wk^Read_59/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_60/DisableCopyOnReadDisableCopyOnRead2read_60_disablecopyonread_adam_m_self_attention_wv"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp2read_60_disablecopyonread_adam_m_self_attention_wv^Read_60/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_61/DisableCopyOnReadDisableCopyOnRead2read_61_disablecopyonread_adam_v_self_attention_wv"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp2read_61_disablecopyonread_adam_v_self_attention_wv^Read_61/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_62/DisableCopyOnReadDisableCopyOnRead-read_62_disablecopyonread_adam_m_att_layer2_w"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp-read_62_disablecopyonread_adam_m_att_layer2_w^Read_62/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_63/DisableCopyOnReadDisableCopyOnRead-read_63_disablecopyonread_adam_v_att_layer2_w"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp-read_63_disablecopyonread_adam_v_att_layer2_w^Read_63/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_64/DisableCopyOnReadDisableCopyOnRead-read_64_disablecopyonread_adam_m_att_layer2_b"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp-read_64_disablecopyonread_adam_m_att_layer2_b^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_65/DisableCopyOnReadDisableCopyOnRead-read_65_disablecopyonread_adam_v_att_layer2_b"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp-read_65_disablecopyonread_adam_v_att_layer2_b^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_66/DisableCopyOnReadDisableCopyOnRead-read_66_disablecopyonread_adam_m_att_layer2_q"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp-read_66_disablecopyonread_adam_m_att_layer2_q^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_67/DisableCopyOnReadDisableCopyOnRead-read_67_disablecopyonread_adam_v_att_layer2_q"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp-read_67_disablecopyonread_adam_v_att_layer2_q^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
:	�t
Read_68/DisableCopyOnReadDisableCopyOnReadread_68_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOpread_68_disablecopyonread_total^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_69/DisableCopyOnReadDisableCopyOnReadread_69_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOpread_69_disablecopyonread_count^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_70/DisableCopyOnReadDisableCopyOnRead(read_70_disablecopyonread_true_positives"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp(read_70_disablecopyonread_true_positives^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_71/DisableCopyOnReadDisableCopyOnRead(read_71_disablecopyonread_true_negatives"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOp(read_71_disablecopyonread_true_negatives^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_72/DisableCopyOnReadDisableCopyOnRead)read_72_disablecopyonread_false_positives"/device:CPU:0*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOp)read_72_disablecopyonread_false_positives^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_73/DisableCopyOnReadDisableCopyOnRead)read_73_disablecopyonread_false_negatives"/device:CPU:0*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOp)read_73_disablecopyonread_false_negatives^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*�
value�B�KB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*�
value�B�KB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *Y
dtypesO
M2K	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_148Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_149IdentityIdentity_148:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_149Identity_149:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=K9

_output_shapes
: 

_user_specified_nameConst:/J+
)
_user_specified_namefalse_negatives:/I+
)
_user_specified_namefalse_positives:.H*
(
_user_specified_nametrue_negatives:.G*
(
_user_specified_nametrue_positives:%F!

_user_specified_namecount:%E!

_user_specified_nametotal:3D/
-
_user_specified_nameAdam/v/att_layer2/q:3C/
-
_user_specified_nameAdam/m/att_layer2/q:3B/
-
_user_specified_nameAdam/v/att_layer2/b:3A/
-
_user_specified_nameAdam/m/att_layer2/b:3@/
-
_user_specified_nameAdam/v/att_layer2/W:3?/
-
_user_specified_nameAdam/m/att_layer2/W:8>4
2
_user_specified_nameAdam/v/self_attention/WV:8=4
2
_user_specified_nameAdam/m/self_attention/WV:8<4
2
_user_specified_nameAdam/v/self_attention/WK:8;4
2
_user_specified_nameAdam/m/self_attention/WK:8:4
2
_user_specified_nameAdam/v/self_attention/WQ:894
2
_user_specified_nameAdam/m/self_attention/WQ:38/
-
_user_specified_nameAdam/v/dense_3/bias:37/
-
_user_specified_nameAdam/m/dense_3/bias:561
/
_user_specified_nameAdam/v/dense_3/kernel:551
/
_user_specified_nameAdam/m/dense_3/kernel:A4=
;
_user_specified_name#!Adam/v/batch_normalization_2/beta:A3=
;
_user_specified_name#!Adam/m/batch_normalization_2/beta:B2>
<
_user_specified_name$"Adam/v/batch_normalization_2/gamma:B1>
<
_user_specified_name$"Adam/m/batch_normalization_2/gamma:30/
-
_user_specified_nameAdam/v/dense_2/bias:3//
-
_user_specified_nameAdam/m/dense_2/bias:5.1
/
_user_specified_nameAdam/v/dense_2/kernel:5-1
/
_user_specified_nameAdam/m/dense_2/kernel:A,=
;
_user_specified_name#!Adam/v/batch_normalization_1/beta:A+=
;
_user_specified_name#!Adam/m/batch_normalization_1/beta:B*>
<
_user_specified_name$"Adam/v/batch_normalization_1/gamma:B)>
<
_user_specified_name$"Adam/m/batch_normalization_1/gamma:3(/
-
_user_specified_nameAdam/v/dense_1/bias:3'/
-
_user_specified_nameAdam/m/dense_1/bias:5&1
/
_user_specified_nameAdam/v/dense_1/kernel:5%1
/
_user_specified_nameAdam/m/dense_1/kernel:?$;
9
_user_specified_name!Adam/v/batch_normalization/beta:?#;
9
_user_specified_name!Adam/m/batch_normalization/beta:@"<
:
_user_specified_name" Adam/v/batch_normalization/gamma:@!<
:
_user_specified_name" Adam/m/batch_normalization/gamma:1 -
+
_user_specified_nameAdam/v/dense/bias:1-
+
_user_specified_nameAdam/m/dense/bias:3/
-
_user_specified_nameAdam/v/dense/kernel:3/
-
_user_specified_nameAdam/m/dense/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_nameatt_layer2/q:,(
&
_user_specified_nameatt_layer2/b:,(
&
_user_specified_nameatt_layer2/W:1-
+
_user_specified_nameself_attention/WV:1-
+
_user_specified_nameself_attention/WK:1-
+
_user_specified_nameself_attention/WQ:EA
?
_user_specified_name'%batch_normalization_2/moving_variance:A=
;
_user_specified_name#!batch_normalization_2/moving_mean:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:A=
;
_user_specified_name#!batch_normalization_1/moving_mean:C?
=
_user_specified_name%#batch_normalization/moving_variance:?;
9
_user_specified_name!batch_normalization/moving_mean:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel::6
4
_user_specified_namebatch_normalization_2/beta:;7
5
_user_specified_namebatch_normalization_2/gamma:,
(
&
_user_specified_namedense_2/bias:.	*
(
_user_specified_namedense_2/kernel::6
4
_user_specified_namebatch_normalization_1/beta:;7
5
_user_specified_namebatch_normalization_1/gamma:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:84
2
_user_specified_namebatch_normalization/beta:95
3
_user_specified_namebatch_normalization/gamma:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
j
>__inference_dot_layer_call_and_return_conditional_losses_85563
inputs_0
inputs_1
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :r

ExpandDims
ExpandDimsinputs_1ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������u
MatMulBatchMatMulV2inputs_0ExpandDims:output:0*
T0*4
_output_shapes"
 :������������������R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::��~
SqueezeSqueezeMatMul:output:0*
T0*0
_output_shapes
:������������������*
squeeze_dims

���������a
IdentityIdentitySqueeze:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:�������������������:����������:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_1:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs_0
�&
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_83062

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_user_encoder_layer_call_fn_84564
input_5
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:
��

unknown_20:
��

unknown_21:
��

unknown_22:
��

unknown_23:	�

unknown_24:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*6
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *P
fKRI
G__inference_user_encoder_layer_call_and_return_conditional_losses_84325p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:���������#�: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name84560:%!

_user_specified_name84558:%!

_user_specified_name84556:%!

_user_specified_name84554:%!

_user_specified_name84552:%!

_user_specified_name84550:%!

_user_specified_name84548:%!

_user_specified_name84546:%!

_user_specified_name84544:%!

_user_specified_name84542:%!

_user_specified_name84540:%!

_user_specified_name84538:%!

_user_specified_name84536:%!

_user_specified_name84534:%!

_user_specified_name84532:%!

_user_specified_name84530:%
!

_user_specified_name84528:%	!

_user_specified_name84526:%!

_user_specified_name84524:%!

_user_specified_name84522:%!

_user_specified_name84520:%!

_user_specified_name84518:%!

_user_specified_name84516:%!

_user_specified_name84514:%!

_user_specified_name84512:%!

_user_specified_name84510:U Q
,
_output_shapes
:���������#�
!
_user_specified_name	input_5
�
O
#__inference_dot_layer_call_fn_85553
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *G
fBR@
>__inference_dot_layer_call_and_return_conditional_losses_84867i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:�������������������:����������:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_1:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs_0
�
�
%__inference_model_layer_call_fn_85134
input_1
input_2
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:
��

unknown_20:
��

unknown_21:
��

unknown_22:
��

unknown_23:	�

unknown_24:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������*<
_read_only_resource_inputs
	
*<
config_proto,*

CPU

GPU 2J 8R(���������� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_85018x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:���������#�:�������������������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name85130:%!

_user_specified_name85128:%!

_user_specified_name85126:%!

_user_specified_name85124:%!

_user_specified_name85122:%!

_user_specified_name85120:%!

_user_specified_name85118:%!

_user_specified_name85116:%!

_user_specified_name85114:%!

_user_specified_name85112:%!

_user_specified_name85110:%!

_user_specified_name85108:%!

_user_specified_name85106:%!

_user_specified_name85104:%!

_user_specified_name85102:%!

_user_specified_name85100:%!

_user_specified_name85098:%
!

_user_specified_name85096:%	!

_user_specified_name85094:%!

_user_specified_name85092:%!

_user_specified_name85090:%!

_user_specified_name85088:%!

_user_specified_name85086:%!

_user_specified_name85084:%!

_user_specified_name85082:%!

_user_specified_name85080:^Z
5
_output_shapes#
!:�������������������
!
_user_specified_name	input_2:U Q
,
_output_shapes
:���������#�
!
_user_specified_name	input_1
�
�
,__inference_news_encoder_layer_call_fn_83481
input_4
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
��

unknown_18:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*6
_read_only_resource_inputs
	
*<
config_proto,*

CPU

GPU 2J 8R(���������� *P
fKRI
G__inference_news_encoder_layer_call_and_return_conditional_losses_83391p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name83477:%!

_user_specified_name83475:%!

_user_specified_name83473:%!

_user_specified_name83471:%!

_user_specified_name83469:%!

_user_specified_name83467:%!

_user_specified_name83465:%!

_user_specified_name83463:%!

_user_specified_name83461:%!

_user_specified_name83459:%
!

_user_specified_name83457:%	!

_user_specified_name83455:%!

_user_specified_name83453:%!

_user_specified_name83451:%!

_user_specified_name83449:%!

_user_specified_name83447:%!

_user_specified_name83445:%!

_user_specified_name83443:%!

_user_specified_name83441:%!

_user_specified_name83439:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_4
�
�
2__inference_time_distributed_1_layer_call_fn_85251

inputs
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
��

unknown_18:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*0
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_83603}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�������������������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name85247:%!

_user_specified_name85245:%!

_user_specified_name85243:%!

_user_specified_name85241:%!

_user_specified_name85239:%!

_user_specified_name85237:%!

_user_specified_name85235:%!

_user_specified_name85233:%!

_user_specified_name85231:%!

_user_specified_name85229:%
!

_user_specified_name85227:%	!

_user_specified_name85225:%!

_user_specified_name85223:%!

_user_specified_name85221:%!

_user_specified_name85219:%!

_user_specified_name85217:%!

_user_specified_name85215:%!

_user_specified_name85213:%!

_user_specified_name85211:%!

_user_specified_name85209:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
��
�I
 __inference__wrapped_model_82948
input_1
input_2^
Jmodel_time_distributed_1_news_encoder_dense_matmul_readvariableop_resource:
��Z
Kmodel_time_distributed_1_news_encoder_dense_biasadd_readvariableop_resource:	�j
[model_time_distributed_1_news_encoder_batch_normalization_batchnorm_readvariableop_resource:	�n
_model_time_distributed_1_news_encoder_batch_normalization_batchnorm_mul_readvariableop_resource:	�l
]model_time_distributed_1_news_encoder_batch_normalization_batchnorm_readvariableop_1_resource:	�l
]model_time_distributed_1_news_encoder_batch_normalization_batchnorm_readvariableop_2_resource:	�`
Lmodel_time_distributed_1_news_encoder_dense_1_matmul_readvariableop_resource:
��\
Mmodel_time_distributed_1_news_encoder_dense_1_biasadd_readvariableop_resource:	�l
]model_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_readvariableop_resource:	�p
amodel_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_mul_readvariableop_resource:	�n
_model_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_readvariableop_1_resource:	�n
_model_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_readvariableop_2_resource:	�`
Lmodel_time_distributed_1_news_encoder_dense_2_matmul_readvariableop_resource:
��\
Mmodel_time_distributed_1_news_encoder_dense_2_biasadd_readvariableop_resource:	�l
]model_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_readvariableop_resource:	�p
amodel_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_mul_readvariableop_resource:	�n
_model_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_readvariableop_1_resource:	�n
_model_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_readvariableop_2_resource:	�`
Lmodel_time_distributed_1_news_encoder_dense_3_matmul_readvariableop_resource:
��\
Mmodel_time_distributed_1_news_encoder_dense_3_biasadd_readvariableop_resource:	�U
Amodel_user_encoder_self_attention_shape_1_readvariableop_resource:
��U
Amodel_user_encoder_self_attention_shape_4_readvariableop_resource:
��U
Amodel_user_encoder_self_attention_shape_7_readvariableop_resource:
��Q
=model_user_encoder_att_layer2_shape_1_readvariableop_resource:
��H
9model_user_encoder_att_layer2_add_readvariableop_resource:	�P
=model_user_encoder_att_layer2_shape_3_readvariableop_resource:	�
identity��Emodel/time_distributed_1/batch_normalization/batchnorm/ReadVariableOp�Gmodel/time_distributed_1/batch_normalization/batchnorm/ReadVariableOp_1�Gmodel/time_distributed_1/batch_normalization/batchnorm/ReadVariableOp_2�Imodel/time_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOp�Gmodel/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp�Imodel/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_1�Imodel/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_2�Kmodel/time_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOp�Gmodel/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp�Imodel/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_1�Imodel/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_2�Kmodel/time_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOp�5model/time_distributed_1/dense/BiasAdd/ReadVariableOp�4model/time_distributed_1/dense/MatMul/ReadVariableOp�7model/time_distributed_1/dense_1/BiasAdd/ReadVariableOp�6model/time_distributed_1/dense_1/MatMul/ReadVariableOp�7model/time_distributed_1/dense_2/BiasAdd/ReadVariableOp�6model/time_distributed_1/dense_2/MatMul/ReadVariableOp�7model/time_distributed_1/dense_3/BiasAdd/ReadVariableOp�6model/time_distributed_1/dense_3/MatMul/ReadVariableOp�Rmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOp�Tmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOp_1�Tmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOp_2�Vmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp�Tmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp�Vmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1�Vmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2�Xmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp�Tmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp�Vmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1�Vmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2�Xmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp�Bmodel/time_distributed_1/news_encoder/dense/BiasAdd/ReadVariableOp�Amodel/time_distributed_1/news_encoder/dense/MatMul/ReadVariableOp�Dmodel/time_distributed_1/news_encoder/dense_1/BiasAdd/ReadVariableOp�Cmodel/time_distributed_1/news_encoder/dense_1/MatMul/ReadVariableOp�Dmodel/time_distributed_1/news_encoder/dense_2/BiasAdd/ReadVariableOp�Cmodel/time_distributed_1/news_encoder/dense_2/MatMul/ReadVariableOp�Dmodel/time_distributed_1/news_encoder/dense_3/BiasAdd/ReadVariableOp�Cmodel/time_distributed_1/news_encoder/dense_3/MatMul/ReadVariableOp�0model/user_encoder/att_layer2/add/ReadVariableOp�6model/user_encoder/att_layer2/transpose/ReadVariableOp�8model/user_encoder/att_layer2/transpose_1/ReadVariableOp�:model/user_encoder/self_attention/transpose/ReadVariableOp�<model/user_encoder/self_attention/transpose_2/ReadVariableOp�<model/user_encoder/self_attention/transpose_4/ReadVariableOp�Pmodel/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOp�Rmodel/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOp_1�Rmodel/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOp_2�Tmodel/user_encoder/time_distributed/batch_normalization/batchnorm/mul/ReadVariableOp�Rmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOp�Tmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOp_1�Tmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOp_2�Vmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOp�Rmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOp�Tmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOp_1�Tmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOp_2�Vmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOp�@model/user_encoder/time_distributed/dense/BiasAdd/ReadVariableOp�?model/user_encoder/time_distributed/dense/MatMul/ReadVariableOp�Bmodel/user_encoder/time_distributed/dense_1/BiasAdd/ReadVariableOp�Amodel/user_encoder/time_distributed/dense_1/MatMul/ReadVariableOp�Bmodel/user_encoder/time_distributed/dense_2/BiasAdd/ReadVariableOp�Amodel/user_encoder/time_distributed/dense_2/MatMul/ReadVariableOp�Bmodel/user_encoder/time_distributed/dense_3/BiasAdd/ReadVariableOp�Amodel/user_encoder/time_distributed/dense_3/MatMul/ReadVariableOp�]model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp�_model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp_1�_model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp_2�amodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp�_model/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp�amodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1�amodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2�cmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp�_model/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp�amodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1�amodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2�cmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp�Mmodel/user_encoder/time_distributed/news_encoder/dense/BiasAdd/ReadVariableOp�Lmodel/user_encoder/time_distributed/news_encoder/dense/MatMul/ReadVariableOp�Omodel/user_encoder/time_distributed/news_encoder/dense_1/BiasAdd/ReadVariableOp�Nmodel/user_encoder/time_distributed/news_encoder/dense_1/MatMul/ReadVariableOp�Omodel/user_encoder/time_distributed/news_encoder/dense_2/BiasAdd/ReadVariableOp�Nmodel/user_encoder/time_distributed/news_encoder/dense_2/MatMul/ReadVariableOp�Omodel/user_encoder/time_distributed/news_encoder/dense_3/BiasAdd/ReadVariableOp�Nmodel/user_encoder/time_distributed/news_encoder/dense_3/MatMul/ReadVariableOpc
model/time_distributed_1/ShapeShapeinput_2*
T0*
_output_shapes
::��v
,model/time_distributed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.model/time_distributed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/time_distributed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&model/time_distributed_1/strided_sliceStridedSlice'model/time_distributed_1/Shape:output:05model/time_distributed_1/strided_slice/stack:output:07model/time_distributed_1/strided_slice/stack_1:output:07model/time_distributed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
&model/time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
 model/time_distributed_1/ReshapeReshapeinput_2/model/time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
Amodel/time_distributed_1/news_encoder/dense/MatMul/ReadVariableOpReadVariableOpJmodel_time_distributed_1_news_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
2model/time_distributed_1/news_encoder/dense/MatMulMatMul)model/time_distributed_1/Reshape:output:0Imodel/time_distributed_1/news_encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Bmodel/time_distributed_1/news_encoder/dense/BiasAdd/ReadVariableOpReadVariableOpKmodel_time_distributed_1_news_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
3model/time_distributed_1/news_encoder/dense/BiasAddBiasAdd<model/time_distributed_1/news_encoder/dense/MatMul:product:0Jmodel/time_distributed_1/news_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
0model/time_distributed_1/news_encoder/dense/ReluRelu<model/time_distributed_1/news_encoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Rmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOpReadVariableOp[model_time_distributed_1_news_encoder_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Imodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Gmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/addAddV2Zmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOp:value:0Rmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
Imodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/RsqrtRsqrtKmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Vmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/mulMulMmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/Rsqrt:y:0^model/time_distributed_1/news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
Imodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/mul_1Mul>model/time_distributed_1/news_encoder/dense/Relu:activations:0Kmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Tmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp]model_time_distributed_1_news_encoder_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Imodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/mul_2Mul\model/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOp_1:value:0Kmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
Tmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp]model_time_distributed_1_news_encoder_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
Gmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/subSub\model/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOp_2:value:0Mmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
Imodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/add_1AddV2Mmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/mul_1:z:0Kmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
6model/time_distributed_1/news_encoder/dropout/IdentityIdentityMmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
Cmodel/time_distributed_1/news_encoder/dense_1/MatMul/ReadVariableOpReadVariableOpLmodel_time_distributed_1_news_encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
4model/time_distributed_1/news_encoder/dense_1/MatMulMatMul?model/time_distributed_1/news_encoder/dropout/Identity:output:0Kmodel/time_distributed_1/news_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Dmodel/time_distributed_1/news_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOpMmodel_time_distributed_1_news_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5model/time_distributed_1/news_encoder/dense_1/BiasAddBiasAdd>model/time_distributed_1/news_encoder/dense_1/MatMul:product:0Lmodel/time_distributed_1/news_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2model/time_distributed_1/news_encoder/dense_1/ReluRelu>model/time_distributed_1/news_encoder/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Tmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp]model_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Kmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Imodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/addAddV2\model/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp:value:0Tmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
Kmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/RsqrtRsqrtMmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Xmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpamodel_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Imodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/mulMulOmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/Rsqrt:y:0`model/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
Kmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/mul_1Mul@model/time_distributed_1/news_encoder/dense_1/Relu:activations:0Mmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Vmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Kmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/mul_2Mul^model/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0Mmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
Vmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
Imodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/subSub^model/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0Omodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
Kmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/add_1AddV2Omodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/mul_1:z:0Mmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
8model/time_distributed_1/news_encoder/dropout_1/IdentityIdentityOmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
Cmodel/time_distributed_1/news_encoder/dense_2/MatMul/ReadVariableOpReadVariableOpLmodel_time_distributed_1_news_encoder_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
4model/time_distributed_1/news_encoder/dense_2/MatMulMatMulAmodel/time_distributed_1/news_encoder/dropout_1/Identity:output:0Kmodel/time_distributed_1/news_encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Dmodel/time_distributed_1/news_encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOpMmodel_time_distributed_1_news_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5model/time_distributed_1/news_encoder/dense_2/BiasAddBiasAdd>model/time_distributed_1/news_encoder/dense_2/MatMul:product:0Lmodel/time_distributed_1/news_encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2model/time_distributed_1/news_encoder/dense_2/ReluRelu>model/time_distributed_1/news_encoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Tmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp]model_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Kmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Imodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/addAddV2\model/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp:value:0Tmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
Kmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/RsqrtRsqrtMmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Xmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpamodel_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Imodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/mulMulOmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/Rsqrt:y:0`model/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
Kmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/mul_1Mul@model/time_distributed_1/news_encoder/dense_2/Relu:activations:0Mmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Vmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Kmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/mul_2Mul^model/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0Mmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
Vmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
Imodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/subSub^model/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0Omodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
Kmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/add_1AddV2Omodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/mul_1:z:0Mmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
8model/time_distributed_1/news_encoder/dropout_2/IdentityIdentityOmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
Cmodel/time_distributed_1/news_encoder/dense_3/MatMul/ReadVariableOpReadVariableOpLmodel_time_distributed_1_news_encoder_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
4model/time_distributed_1/news_encoder/dense_3/MatMulMatMulAmodel/time_distributed_1/news_encoder/dropout_2/Identity:output:0Kmodel/time_distributed_1/news_encoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Dmodel/time_distributed_1/news_encoder/dense_3/BiasAdd/ReadVariableOpReadVariableOpMmodel_time_distributed_1_news_encoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5model/time_distributed_1/news_encoder/dense_3/BiasAddBiasAdd>model/time_distributed_1/news_encoder/dense_3/MatMul:product:0Lmodel/time_distributed_1/news_encoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2model/time_distributed_1/news_encoder/dense_3/ReluRelu>model/time_distributed_1/news_encoder/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������u
*model/time_distributed_1/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������m
*model/time_distributed_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
(model/time_distributed_1/Reshape_1/shapePack3model/time_distributed_1/Reshape_1/shape/0:output:0/model/time_distributed_1/strided_slice:output:03model/time_distributed_1/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
"model/time_distributed_1/Reshape_1Reshape@model/time_distributed_1/news_encoder/dense_3/Relu:activations:01model/time_distributed_1/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������y
(model/time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"model/time_distributed_1/Reshape_2Reshapeinput_21model/time_distributed_1/Reshape_2/shape:output:0*
T0*(
_output_shapes
:�����������
4model/time_distributed_1/dense/MatMul/ReadVariableOpReadVariableOpJmodel_time_distributed_1_news_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
%model/time_distributed_1/dense/MatMulMatMul+model/time_distributed_1/Reshape_2:output:0<model/time_distributed_1/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
5model/time_distributed_1/dense/BiasAdd/ReadVariableOpReadVariableOpKmodel_time_distributed_1_news_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&model/time_distributed_1/dense/BiasAddBiasAdd/model/time_distributed_1/dense/MatMul:product:0=model/time_distributed_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#model/time_distributed_1/dense/ReluRelu/model/time_distributed_1/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Emodel/time_distributed_1/batch_normalization/batchnorm/ReadVariableOpReadVariableOp[model_time_distributed_1_news_encoder_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<model/time_distributed_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
:model/time_distributed_1/batch_normalization/batchnorm/addAddV2Mmodel/time_distributed_1/batch_normalization/batchnorm/ReadVariableOp:value:0Emodel/time_distributed_1/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
<model/time_distributed_1/batch_normalization/batchnorm/RsqrtRsqrt>model/time_distributed_1/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Imodel/time_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:model/time_distributed_1/batch_normalization/batchnorm/mulMul@model/time_distributed_1/batch_normalization/batchnorm/Rsqrt:y:0Qmodel/time_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
<model/time_distributed_1/batch_normalization/batchnorm/mul_1Mul1model/time_distributed_1/dense/Relu:activations:0>model/time_distributed_1/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Gmodel/time_distributed_1/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp]model_time_distributed_1_news_encoder_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
<model/time_distributed_1/batch_normalization/batchnorm/mul_2MulOmodel/time_distributed_1/batch_normalization/batchnorm/ReadVariableOp_1:value:0>model/time_distributed_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
Gmodel/time_distributed_1/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp]model_time_distributed_1_news_encoder_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
:model/time_distributed_1/batch_normalization/batchnorm/subSubOmodel/time_distributed_1/batch_normalization/batchnorm/ReadVariableOp_2:value:0@model/time_distributed_1/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
<model/time_distributed_1/batch_normalization/batchnorm/add_1AddV2@model/time_distributed_1/batch_normalization/batchnorm/mul_1:z:0>model/time_distributed_1/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
)model/time_distributed_1/dropout/IdentityIdentity@model/time_distributed_1/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
6model/time_distributed_1/dense_1/MatMul/ReadVariableOpReadVariableOpLmodel_time_distributed_1_news_encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'model/time_distributed_1/dense_1/MatMulMatMul2model/time_distributed_1/dropout/Identity:output:0>model/time_distributed_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
7model/time_distributed_1/dense_1/BiasAdd/ReadVariableOpReadVariableOpMmodel_time_distributed_1_news_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(model/time_distributed_1/dense_1/BiasAddBiasAdd1model/time_distributed_1/dense_1/MatMul:product:0?model/time_distributed_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%model/time_distributed_1/dense_1/ReluRelu1model/time_distributed_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Gmodel/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp]model_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
>model/time_distributed_1/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<model/time_distributed_1/batch_normalization_1/batchnorm/addAddV2Omodel/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp:value:0Gmodel/time_distributed_1/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
>model/time_distributed_1/batch_normalization_1/batchnorm/RsqrtRsqrt@model/time_distributed_1/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Kmodel/time_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpamodel_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<model/time_distributed_1/batch_normalization_1/batchnorm/mulMulBmodel/time_distributed_1/batch_normalization_1/batchnorm/Rsqrt:y:0Smodel/time_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
>model/time_distributed_1/batch_normalization_1/batchnorm/mul_1Mul3model/time_distributed_1/dense_1/Relu:activations:0@model/time_distributed_1/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Imodel/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
>model/time_distributed_1/batch_normalization_1/batchnorm/mul_2MulQmodel/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0@model/time_distributed_1/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
Imodel/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
<model/time_distributed_1/batch_normalization_1/batchnorm/subSubQmodel/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0Bmodel/time_distributed_1/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
>model/time_distributed_1/batch_normalization_1/batchnorm/add_1AddV2Bmodel/time_distributed_1/batch_normalization_1/batchnorm/mul_1:z:0@model/time_distributed_1/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
+model/time_distributed_1/dropout_1/IdentityIdentityBmodel/time_distributed_1/batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
6model/time_distributed_1/dense_2/MatMul/ReadVariableOpReadVariableOpLmodel_time_distributed_1_news_encoder_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'model/time_distributed_1/dense_2/MatMulMatMul4model/time_distributed_1/dropout_1/Identity:output:0>model/time_distributed_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
7model/time_distributed_1/dense_2/BiasAdd/ReadVariableOpReadVariableOpMmodel_time_distributed_1_news_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(model/time_distributed_1/dense_2/BiasAddBiasAdd1model/time_distributed_1/dense_2/MatMul:product:0?model/time_distributed_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%model/time_distributed_1/dense_2/ReluRelu1model/time_distributed_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Gmodel/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp]model_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
>model/time_distributed_1/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<model/time_distributed_1/batch_normalization_2/batchnorm/addAddV2Omodel/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp:value:0Gmodel/time_distributed_1/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
>model/time_distributed_1/batch_normalization_2/batchnorm/RsqrtRsqrt@model/time_distributed_1/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Kmodel/time_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpamodel_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<model/time_distributed_1/batch_normalization_2/batchnorm/mulMulBmodel/time_distributed_1/batch_normalization_2/batchnorm/Rsqrt:y:0Smodel/time_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
>model/time_distributed_1/batch_normalization_2/batchnorm/mul_1Mul3model/time_distributed_1/dense_2/Relu:activations:0@model/time_distributed_1/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Imodel/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
>model/time_distributed_1/batch_normalization_2/batchnorm/mul_2MulQmodel/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0@model/time_distributed_1/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
Imodel/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
<model/time_distributed_1/batch_normalization_2/batchnorm/subSubQmodel/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0Bmodel/time_distributed_1/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
>model/time_distributed_1/batch_normalization_2/batchnorm/add_1AddV2Bmodel/time_distributed_1/batch_normalization_2/batchnorm/mul_1:z:0@model/time_distributed_1/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
+model/time_distributed_1/dropout_2/IdentityIdentityBmodel/time_distributed_1/batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
6model/time_distributed_1/dense_3/MatMul/ReadVariableOpReadVariableOpLmodel_time_distributed_1_news_encoder_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'model/time_distributed_1/dense_3/MatMulMatMul4model/time_distributed_1/dropout_2/Identity:output:0>model/time_distributed_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
7model/time_distributed_1/dense_3/BiasAdd/ReadVariableOpReadVariableOpMmodel_time_distributed_1_news_encoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(model/time_distributed_1/dense_3/BiasAddBiasAdd1model/time_distributed_1/dense_3/MatMul:product:0?model/time_distributed_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%model/time_distributed_1/dense_3/ReluRelu1model/time_distributed_1/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
1model/user_encoder/time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
+model/user_encoder/time_distributed/ReshapeReshapeinput_1:model/user_encoder/time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
Lmodel/user_encoder/time_distributed/news_encoder/dense/MatMul/ReadVariableOpReadVariableOpJmodel_time_distributed_1_news_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
=model/user_encoder/time_distributed/news_encoder/dense/MatMulMatMul4model/user_encoder/time_distributed/Reshape:output:0Tmodel/user_encoder/time_distributed/news_encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Mmodel/user_encoder/time_distributed/news_encoder/dense/BiasAdd/ReadVariableOpReadVariableOpKmodel_time_distributed_1_news_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
>model/user_encoder/time_distributed/news_encoder/dense/BiasAddBiasAddGmodel/user_encoder/time_distributed/news_encoder/dense/MatMul:product:0Umodel/user_encoder/time_distributed/news_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
;model/user_encoder/time_distributed/news_encoder/dense/ReluReluGmodel/user_encoder/time_distributed/news_encoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
]model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOpReadVariableOp[model_time_distributed_1_news_encoder_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Tmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Rmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/addAddV2emodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp:value:0]model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
Tmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/RsqrtRsqrtVmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
amodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Rmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/mulMulXmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/Rsqrt:y:0imodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
Tmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/mul_1MulImodel/user_encoder/time_distributed/news_encoder/dense/Relu:activations:0Vmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
_model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp]model_time_distributed_1_news_encoder_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Tmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/mul_2Mulgmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp_1:value:0Vmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
_model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp]model_time_distributed_1_news_encoder_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
Rmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/subSubgmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp_2:value:0Xmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
Tmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/add_1AddV2Xmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/mul_1:z:0Vmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
Amodel/user_encoder/time_distributed/news_encoder/dropout/IdentityIdentityXmodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
Nmodel/user_encoder/time_distributed/news_encoder/dense_1/MatMul/ReadVariableOpReadVariableOpLmodel_time_distributed_1_news_encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
?model/user_encoder/time_distributed/news_encoder/dense_1/MatMulMatMulJmodel/user_encoder/time_distributed/news_encoder/dropout/Identity:output:0Vmodel/user_encoder/time_distributed/news_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Omodel/user_encoder/time_distributed/news_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOpMmodel_time_distributed_1_news_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
@model/user_encoder/time_distributed/news_encoder/dense_1/BiasAddBiasAddImodel/user_encoder/time_distributed/news_encoder/dense_1/MatMul:product:0Wmodel/user_encoder/time_distributed/news_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
=model/user_encoder/time_distributed/news_encoder/dense_1/ReluReluImodel/user_encoder/time_distributed/news_encoder/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
_model/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp]model_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Vmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Tmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/addAddV2gmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp:value:0_model/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
Vmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/RsqrtRsqrtXmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
cmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpamodel_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Tmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/mulMulZmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/Rsqrt:y:0kmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
Vmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/mul_1MulKmodel/user_encoder/time_distributed/news_encoder/dense_1/Relu:activations:0Xmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
amodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Vmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/mul_2Mulimodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0Xmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
amodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
Tmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/subSubimodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0Zmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
Vmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/add_1AddV2Zmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/mul_1:z:0Xmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
Cmodel/user_encoder/time_distributed/news_encoder/dropout_1/IdentityIdentityZmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
Nmodel/user_encoder/time_distributed/news_encoder/dense_2/MatMul/ReadVariableOpReadVariableOpLmodel_time_distributed_1_news_encoder_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
?model/user_encoder/time_distributed/news_encoder/dense_2/MatMulMatMulLmodel/user_encoder/time_distributed/news_encoder/dropout_1/Identity:output:0Vmodel/user_encoder/time_distributed/news_encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Omodel/user_encoder/time_distributed/news_encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOpMmodel_time_distributed_1_news_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
@model/user_encoder/time_distributed/news_encoder/dense_2/BiasAddBiasAddImodel/user_encoder/time_distributed/news_encoder/dense_2/MatMul:product:0Wmodel/user_encoder/time_distributed/news_encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
=model/user_encoder/time_distributed/news_encoder/dense_2/ReluReluImodel/user_encoder/time_distributed/news_encoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
_model/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp]model_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Vmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Tmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/addAddV2gmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp:value:0_model/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
Vmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/RsqrtRsqrtXmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
cmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpamodel_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Tmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/mulMulZmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/Rsqrt:y:0kmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
Vmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/mul_1MulKmodel/user_encoder/time_distributed/news_encoder/dense_2/Relu:activations:0Xmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
amodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Vmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/mul_2Mulimodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0Xmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
amodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
Tmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/subSubimodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0Zmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
Vmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/add_1AddV2Zmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/mul_1:z:0Xmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
Cmodel/user_encoder/time_distributed/news_encoder/dropout_2/IdentityIdentityZmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
Nmodel/user_encoder/time_distributed/news_encoder/dense_3/MatMul/ReadVariableOpReadVariableOpLmodel_time_distributed_1_news_encoder_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
?model/user_encoder/time_distributed/news_encoder/dense_3/MatMulMatMulLmodel/user_encoder/time_distributed/news_encoder/dropout_2/Identity:output:0Vmodel/user_encoder/time_distributed/news_encoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Omodel/user_encoder/time_distributed/news_encoder/dense_3/BiasAdd/ReadVariableOpReadVariableOpMmodel_time_distributed_1_news_encoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
@model/user_encoder/time_distributed/news_encoder/dense_3/BiasAddBiasAddImodel/user_encoder/time_distributed/news_encoder/dense_3/MatMul:product:0Wmodel/user_encoder/time_distributed/news_encoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
=model/user_encoder/time_distributed/news_encoder/dense_3/ReluReluImodel/user_encoder/time_distributed/news_encoder/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
3model/user_encoder/time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����#   �  �
-model/user_encoder/time_distributed/Reshape_1ReshapeKmodel/user_encoder/time_distributed/news_encoder/dense_3/Relu:activations:0<model/user_encoder/time_distributed/Reshape_1/shape:output:0*
T0*,
_output_shapes
:���������#��
3model/user_encoder/time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
-model/user_encoder/time_distributed/Reshape_2Reshapeinput_1<model/user_encoder/time_distributed/Reshape_2/shape:output:0*
T0*(
_output_shapes
:�����������
?model/user_encoder/time_distributed/dense/MatMul/ReadVariableOpReadVariableOpJmodel_time_distributed_1_news_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
0model/user_encoder/time_distributed/dense/MatMulMatMul6model/user_encoder/time_distributed/Reshape_2:output:0Gmodel/user_encoder/time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
@model/user_encoder/time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOpKmodel_time_distributed_1_news_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
1model/user_encoder/time_distributed/dense/BiasAddBiasAdd:model/user_encoder/time_distributed/dense/MatMul:product:0Hmodel/user_encoder/time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.model/user_encoder/time_distributed/dense/ReluRelu:model/user_encoder/time_distributed/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Pmodel/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOpReadVariableOp[model_time_distributed_1_news_encoder_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gmodel/user_encoder/time_distributed/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Emodel/user_encoder/time_distributed/batch_normalization/batchnorm/addAddV2Xmodel/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOp:value:0Pmodel/user_encoder/time_distributed/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
Gmodel/user_encoder/time_distributed/batch_normalization/batchnorm/RsqrtRsqrtImodel/user_encoder/time_distributed/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Tmodel/user_encoder/time_distributed/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Emodel/user_encoder/time_distributed/batch_normalization/batchnorm/mulMulKmodel/user_encoder/time_distributed/batch_normalization/batchnorm/Rsqrt:y:0\model/user_encoder/time_distributed/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
Gmodel/user_encoder/time_distributed/batch_normalization/batchnorm/mul_1Mul<model/user_encoder/time_distributed/dense/Relu:activations:0Imodel/user_encoder/time_distributed/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Rmodel/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp]model_time_distributed_1_news_encoder_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Gmodel/user_encoder/time_distributed/batch_normalization/batchnorm/mul_2MulZmodel/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOp_1:value:0Imodel/user_encoder/time_distributed/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
Rmodel/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp]model_time_distributed_1_news_encoder_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
Emodel/user_encoder/time_distributed/batch_normalization/batchnorm/subSubZmodel/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOp_2:value:0Kmodel/user_encoder/time_distributed/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
Gmodel/user_encoder/time_distributed/batch_normalization/batchnorm/add_1AddV2Kmodel/user_encoder/time_distributed/batch_normalization/batchnorm/mul_1:z:0Imodel/user_encoder/time_distributed/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
4model/user_encoder/time_distributed/dropout/IdentityIdentityKmodel/user_encoder/time_distributed/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
Amodel/user_encoder/time_distributed/dense_1/MatMul/ReadVariableOpReadVariableOpLmodel_time_distributed_1_news_encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
2model/user_encoder/time_distributed/dense_1/MatMulMatMul=model/user_encoder/time_distributed/dropout/Identity:output:0Imodel/user_encoder/time_distributed/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Bmodel/user_encoder/time_distributed/dense_1/BiasAdd/ReadVariableOpReadVariableOpMmodel_time_distributed_1_news_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
3model/user_encoder/time_distributed/dense_1/BiasAddBiasAdd<model/user_encoder/time_distributed/dense_1/MatMul:product:0Jmodel/user_encoder/time_distributed/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
0model/user_encoder/time_distributed/dense_1/ReluRelu<model/user_encoder/time_distributed/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Rmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp]model_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Imodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Gmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/addAddV2Zmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOp:value:0Rmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
Imodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/RsqrtRsqrtKmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Vmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpamodel_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/mulMulMmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/Rsqrt:y:0^model/user_encoder/time_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
Imodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/mul_1Mul>model/user_encoder/time_distributed/dense_1/Relu:activations:0Kmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Tmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Imodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/mul_2Mul\model/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0Kmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
Tmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
Gmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/subSub\model/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0Mmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
Imodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/add_1AddV2Mmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/mul_1:z:0Kmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
6model/user_encoder/time_distributed/dropout_1/IdentityIdentityMmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
Amodel/user_encoder/time_distributed/dense_2/MatMul/ReadVariableOpReadVariableOpLmodel_time_distributed_1_news_encoder_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
2model/user_encoder/time_distributed/dense_2/MatMulMatMul?model/user_encoder/time_distributed/dropout_1/Identity:output:0Imodel/user_encoder/time_distributed/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Bmodel/user_encoder/time_distributed/dense_2/BiasAdd/ReadVariableOpReadVariableOpMmodel_time_distributed_1_news_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
3model/user_encoder/time_distributed/dense_2/BiasAddBiasAdd<model/user_encoder/time_distributed/dense_2/MatMul:product:0Jmodel/user_encoder/time_distributed/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
0model/user_encoder/time_distributed/dense_2/ReluRelu<model/user_encoder/time_distributed/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Rmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp]model_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Imodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Gmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/addAddV2Zmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOp:value:0Rmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
Imodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/RsqrtRsqrtKmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Vmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpamodel_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/mulMulMmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/Rsqrt:y:0^model/user_encoder/time_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
Imodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/mul_1Mul>model/user_encoder/time_distributed/dense_2/Relu:activations:0Kmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Tmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Imodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/mul_2Mul\model/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0Kmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
Tmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp_model_time_distributed_1_news_encoder_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
Gmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/subSub\model/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0Mmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
Imodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/add_1AddV2Mmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/mul_1:z:0Kmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
6model/user_encoder/time_distributed/dropout_2/IdentityIdentityMmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
Amodel/user_encoder/time_distributed/dense_3/MatMul/ReadVariableOpReadVariableOpLmodel_time_distributed_1_news_encoder_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
2model/user_encoder/time_distributed/dense_3/MatMulMatMul?model/user_encoder/time_distributed/dropout_2/Identity:output:0Imodel/user_encoder/time_distributed/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Bmodel/user_encoder/time_distributed/dense_3/BiasAdd/ReadVariableOpReadVariableOpMmodel_time_distributed_1_news_encoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
3model/user_encoder/time_distributed/dense_3/BiasAddBiasAdd<model/user_encoder/time_distributed/dense_3/MatMul:product:0Jmodel/user_encoder/time_distributed/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
0model/user_encoder/time_distributed/dense_3/ReluRelu<model/user_encoder/time_distributed/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
'model/user_encoder/self_attention/ShapeShape6model/user_encoder/time_distributed/Reshape_1:output:0*
T0*
_output_shapes
::���
)model/user_encoder/self_attention/unstackUnpack0model/user_encoder/self_attention/Shape:output:0*
T0*
_output_shapes
: : : *	
num�
8model/user_encoder/self_attention/Shape_1/ReadVariableOpReadVariableOpAmodel_user_encoder_self_attention_shape_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0z
)model/user_encoder/self_attention/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"�  �  �
+model/user_encoder/self_attention/unstack_1Unpack2model/user_encoder/self_attention/Shape_1:output:0*
T0*
_output_shapes
: : *	
num�
/model/user_encoder/self_attention/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
)model/user_encoder/self_attention/ReshapeReshape6model/user_encoder/time_distributed/Reshape_1:output:08model/user_encoder/self_attention/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
:model/user_encoder/self_attention/transpose/ReadVariableOpReadVariableOpAmodel_user_encoder_self_attention_shape_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
0model/user_encoder/self_attention/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
+model/user_encoder/self_attention/transpose	TransposeBmodel/user_encoder/self_attention/transpose/ReadVariableOp:value:09model/user_encoder/self_attention/transpose/perm:output:0*
T0* 
_output_shapes
:
���
1model/user_encoder/self_attention/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"�  �����
+model/user_encoder/self_attention/Reshape_1Reshape/model/user_encoder/self_attention/transpose:y:0:model/user_encoder/self_attention/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
(model/user_encoder/self_attention/MatMulMatMul2model/user_encoder/self_attention/Reshape:output:04model/user_encoder/self_attention/Reshape_1:output:0*
T0*(
_output_shapes
:����������u
3model/user_encoder/self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#v
3model/user_encoder/self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
1model/user_encoder/self_attention/Reshape_2/shapePack2model/user_encoder/self_attention/unstack:output:0<model/user_encoder/self_attention/Reshape_2/shape/1:output:0<model/user_encoder/self_attention/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:�
+model/user_encoder/self_attention/Reshape_2Reshape2model/user_encoder/self_attention/MatMul:product:0:model/user_encoder/self_attention/Reshape_2/shape:output:0*
T0*,
_output_shapes
:���������#��
)model/user_encoder/self_attention/Shape_2Shape4model/user_encoder/self_attention/Reshape_2:output:0*
T0*
_output_shapes
::��
5model/user_encoder/self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:�
7model/user_encoder/self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
7model/user_encoder/self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/model/user_encoder/self_attention/strided_sliceStridedSlice2model/user_encoder/self_attention/Shape_2:output:0>model/user_encoder/self_attention/strided_slice/stack:output:0@model/user_encoder/self_attention/strided_slice/stack_1:output:0@model/user_encoder/self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
3model/user_encoder/self_attention/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������u
3model/user_encoder/self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :u
3model/user_encoder/self_attention/Reshape_3/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
1model/user_encoder/self_attention/Reshape_3/shapePack<model/user_encoder/self_attention/Reshape_3/shape/0:output:08model/user_encoder/self_attention/strided_slice:output:0<model/user_encoder/self_attention/Reshape_3/shape/2:output:0<model/user_encoder/self_attention/Reshape_3/shape/3:output:0*
N*
T0*
_output_shapes
:�
+model/user_encoder/self_attention/Reshape_3Reshape4model/user_encoder/self_attention/Reshape_2:output:0:model/user_encoder/self_attention/Reshape_3/shape:output:0*
T0*/
_output_shapes
:���������#�
2model/user_encoder/self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
-model/user_encoder/self_attention/transpose_1	Transpose4model/user_encoder/self_attention/Reshape_3:output:0;model/user_encoder/self_attention/transpose_1/perm:output:0*
T0*/
_output_shapes
:���������#�
)model/user_encoder/self_attention/Shape_3Shape6model/user_encoder/time_distributed/Reshape_1:output:0*
T0*
_output_shapes
::���
+model/user_encoder/self_attention/unstack_2Unpack2model/user_encoder/self_attention/Shape_3:output:0*
T0*
_output_shapes
: : : *	
num�
8model/user_encoder/self_attention/Shape_4/ReadVariableOpReadVariableOpAmodel_user_encoder_self_attention_shape_4_readvariableop_resource* 
_output_shapes
:
��*
dtype0z
)model/user_encoder/self_attention/Shape_4Const*
_output_shapes
:*
dtype0*
valueB"�  �  �
+model/user_encoder/self_attention/unstack_3Unpack2model/user_encoder/self_attention/Shape_4:output:0*
T0*
_output_shapes
: : *	
num�
1model/user_encoder/self_attention/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
+model/user_encoder/self_attention/Reshape_4Reshape6model/user_encoder/time_distributed/Reshape_1:output:0:model/user_encoder/self_attention/Reshape_4/shape:output:0*
T0*(
_output_shapes
:�����������
<model/user_encoder/self_attention/transpose_2/ReadVariableOpReadVariableOpAmodel_user_encoder_self_attention_shape_4_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
2model/user_encoder/self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       �
-model/user_encoder/self_attention/transpose_2	TransposeDmodel/user_encoder/self_attention/transpose_2/ReadVariableOp:value:0;model/user_encoder/self_attention/transpose_2/perm:output:0*
T0* 
_output_shapes
:
���
1model/user_encoder/self_attention/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB"�  �����
+model/user_encoder/self_attention/Reshape_5Reshape1model/user_encoder/self_attention/transpose_2:y:0:model/user_encoder/self_attention/Reshape_5/shape:output:0*
T0* 
_output_shapes
:
���
*model/user_encoder/self_attention/MatMul_1MatMul4model/user_encoder/self_attention/Reshape_4:output:04model/user_encoder/self_attention/Reshape_5:output:0*
T0*(
_output_shapes
:����������u
3model/user_encoder/self_attention/Reshape_6/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#v
3model/user_encoder/self_attention/Reshape_6/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
1model/user_encoder/self_attention/Reshape_6/shapePack4model/user_encoder/self_attention/unstack_2:output:0<model/user_encoder/self_attention/Reshape_6/shape/1:output:0<model/user_encoder/self_attention/Reshape_6/shape/2:output:0*
N*
T0*
_output_shapes
:�
+model/user_encoder/self_attention/Reshape_6Reshape4model/user_encoder/self_attention/MatMul_1:product:0:model/user_encoder/self_attention/Reshape_6/shape:output:0*
T0*,
_output_shapes
:���������#��
)model/user_encoder/self_attention/Shape_5Shape4model/user_encoder/self_attention/Reshape_6:output:0*
T0*
_output_shapes
::���
7model/user_encoder/self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
9model/user_encoder/self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9model/user_encoder/self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1model/user_encoder/self_attention/strided_slice_1StridedSlice2model/user_encoder/self_attention/Shape_5:output:0@model/user_encoder/self_attention/strided_slice_1/stack:output:0Bmodel/user_encoder/self_attention/strided_slice_1/stack_1:output:0Bmodel/user_encoder/self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
3model/user_encoder/self_attention/Reshape_7/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������u
3model/user_encoder/self_attention/Reshape_7/shape/2Const*
_output_shapes
: *
dtype0*
value	B :u
3model/user_encoder/self_attention/Reshape_7/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
1model/user_encoder/self_attention/Reshape_7/shapePack<model/user_encoder/self_attention/Reshape_7/shape/0:output:0:model/user_encoder/self_attention/strided_slice_1:output:0<model/user_encoder/self_attention/Reshape_7/shape/2:output:0<model/user_encoder/self_attention/Reshape_7/shape/3:output:0*
N*
T0*
_output_shapes
:�
+model/user_encoder/self_attention/Reshape_7Reshape4model/user_encoder/self_attention/Reshape_6:output:0:model/user_encoder/self_attention/Reshape_7/shape:output:0*
T0*/
_output_shapes
:���������#�
2model/user_encoder/self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
-model/user_encoder/self_attention/transpose_3	Transpose4model/user_encoder/self_attention/Reshape_7:output:0;model/user_encoder/self_attention/transpose_3/perm:output:0*
T0*/
_output_shapes
:���������#�
)model/user_encoder/self_attention/Shape_6Shape6model/user_encoder/time_distributed/Reshape_1:output:0*
T0*
_output_shapes
::���
+model/user_encoder/self_attention/unstack_4Unpack2model/user_encoder/self_attention/Shape_6:output:0*
T0*
_output_shapes
: : : *	
num�
8model/user_encoder/self_attention/Shape_7/ReadVariableOpReadVariableOpAmodel_user_encoder_self_attention_shape_7_readvariableop_resource* 
_output_shapes
:
��*
dtype0z
)model/user_encoder/self_attention/Shape_7Const*
_output_shapes
:*
dtype0*
valueB"�  �  �
+model/user_encoder/self_attention/unstack_5Unpack2model/user_encoder/self_attention/Shape_7:output:0*
T0*
_output_shapes
: : *	
num�
1model/user_encoder/self_attention/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
+model/user_encoder/self_attention/Reshape_8Reshape6model/user_encoder/time_distributed/Reshape_1:output:0:model/user_encoder/self_attention/Reshape_8/shape:output:0*
T0*(
_output_shapes
:�����������
<model/user_encoder/self_attention/transpose_4/ReadVariableOpReadVariableOpAmodel_user_encoder_self_attention_shape_7_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
2model/user_encoder/self_attention/transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       �
-model/user_encoder/self_attention/transpose_4	TransposeDmodel/user_encoder/self_attention/transpose_4/ReadVariableOp:value:0;model/user_encoder/self_attention/transpose_4/perm:output:0*
T0* 
_output_shapes
:
���
1model/user_encoder/self_attention/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"�  �����
+model/user_encoder/self_attention/Reshape_9Reshape1model/user_encoder/self_attention/transpose_4:y:0:model/user_encoder/self_attention/Reshape_9/shape:output:0*
T0* 
_output_shapes
:
���
*model/user_encoder/self_attention/MatMul_2MatMul4model/user_encoder/self_attention/Reshape_8:output:04model/user_encoder/self_attention/Reshape_9:output:0*
T0*(
_output_shapes
:����������v
4model/user_encoder/self_attention/Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#w
4model/user_encoder/self_attention/Reshape_10/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
2model/user_encoder/self_attention/Reshape_10/shapePack4model/user_encoder/self_attention/unstack_4:output:0=model/user_encoder/self_attention/Reshape_10/shape/1:output:0=model/user_encoder/self_attention/Reshape_10/shape/2:output:0*
N*
T0*
_output_shapes
:�
,model/user_encoder/self_attention/Reshape_10Reshape4model/user_encoder/self_attention/MatMul_2:product:0;model/user_encoder/self_attention/Reshape_10/shape:output:0*
T0*,
_output_shapes
:���������#��
)model/user_encoder/self_attention/Shape_8Shape5model/user_encoder/self_attention/Reshape_10:output:0*
T0*
_output_shapes
::���
7model/user_encoder/self_attention/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
9model/user_encoder/self_attention/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9model/user_encoder/self_attention/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1model/user_encoder/self_attention/strided_slice_2StridedSlice2model/user_encoder/self_attention/Shape_8:output:0@model/user_encoder/self_attention/strided_slice_2/stack:output:0Bmodel/user_encoder/self_attention/strided_slice_2/stack_1:output:0Bmodel/user_encoder/self_attention/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
4model/user_encoder/self_attention/Reshape_11/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������v
4model/user_encoder/self_attention/Reshape_11/shape/2Const*
_output_shapes
: *
dtype0*
value	B :v
4model/user_encoder/self_attention/Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
2model/user_encoder/self_attention/Reshape_11/shapePack=model/user_encoder/self_attention/Reshape_11/shape/0:output:0:model/user_encoder/self_attention/strided_slice_2:output:0=model/user_encoder/self_attention/Reshape_11/shape/2:output:0=model/user_encoder/self_attention/Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:�
,model/user_encoder/self_attention/Reshape_11Reshape5model/user_encoder/self_attention/Reshape_10:output:0;model/user_encoder/self_attention/Reshape_11/shape:output:0*
T0*/
_output_shapes
:���������#�
2model/user_encoder/self_attention/transpose_5/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
-model/user_encoder/self_attention/transpose_5	Transpose5model/user_encoder/self_attention/Reshape_11:output:0;model/user_encoder/self_attention/transpose_5/perm:output:0*
T0*/
_output_shapes
:���������#�
*model/user_encoder/self_attention/MatMul_3BatchMatMulV21model/user_encoder/self_attention/transpose_1:y:01model/user_encoder/self_attention/transpose_3:y:0*
T0*/
_output_shapes
:���������##*
adj_y(j
(model/user_encoder/self_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :�
&model/user_encoder/self_attention/CastCast1model/user_encoder/self_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: l
'model/user_encoder/self_attention/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
)model/user_encoder/self_attention/MaximumMaximum*model/user_encoder/self_attention/Cast:y:00model/user_encoder/self_attention/Const:output:0*
T0*
_output_shapes
: ~
&model/user_encoder/self_attention/SqrtSqrt-model/user_encoder/self_attention/Maximum:z:0*
T0*
_output_shapes
: �
)model/user_encoder/self_attention/truedivRealDiv3model/user_encoder/self_attention/MatMul_3:output:0*model/user_encoder/self_attention/Sqrt:y:0*
T0*/
_output_shapes
:���������##�
2model/user_encoder/self_attention/transpose_6/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
-model/user_encoder/self_attention/transpose_6	Transpose-model/user_encoder/self_attention/truediv:z:0;model/user_encoder/self_attention/transpose_6/perm:output:0*
T0*/
_output_shapes
:���������##�
2model/user_encoder/self_attention/transpose_7/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
-model/user_encoder/self_attention/transpose_7	Transpose1model/user_encoder/self_attention/transpose_6:y:0;model/user_encoder/self_attention/transpose_7/perm:output:0*
T0*/
_output_shapes
:���������##�
)model/user_encoder/self_attention/SoftmaxSoftmax1model/user_encoder/self_attention/transpose_7:y:0*
T0*/
_output_shapes
:���������##�
*model/user_encoder/self_attention/MatMul_4BatchMatMulV23model/user_encoder/self_attention/Softmax:softmax:01model/user_encoder/self_attention/transpose_5:y:0*
T0*/
_output_shapes
:���������#*
adj_x(�
2model/user_encoder/self_attention/transpose_8/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
-model/user_encoder/self_attention/transpose_8	Transpose3model/user_encoder/self_attention/MatMul_4:output:0;model/user_encoder/self_attention/transpose_8/perm:output:0*
T0*/
_output_shapes
:���������#�
)model/user_encoder/self_attention/Shape_9Shape1model/user_encoder/self_attention/transpose_8:y:0*
T0*
_output_shapes
::���
7model/user_encoder/self_attention/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:�
9model/user_encoder/self_attention/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9model/user_encoder/self_attention/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1model/user_encoder/self_attention/strided_slice_3StridedSlice2model/user_encoder/self_attention/Shape_9:output:0@model/user_encoder/self_attention/strided_slice_3/stack:output:0Bmodel/user_encoder/self_attention/strided_slice_3/stack_1:output:0Bmodel/user_encoder/self_attention/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
4model/user_encoder/self_attention/Reshape_12/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������w
4model/user_encoder/self_attention/Reshape_12/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
2model/user_encoder/self_attention/Reshape_12/shapePack=model/user_encoder/self_attention/Reshape_12/shape/0:output:0:model/user_encoder/self_attention/strided_slice_3:output:0=model/user_encoder/self_attention/Reshape_12/shape/2:output:0*
N*
T0*
_output_shapes
:�
,model/user_encoder/self_attention/Reshape_12Reshape1model/user_encoder/self_attention/transpose_8:y:0;model/user_encoder/self_attention/Reshape_12/shape:output:0*
T0*,
_output_shapes
:���������#��
#model/user_encoder/att_layer2/ShapeShape5model/user_encoder/self_attention/Reshape_12:output:0*
T0*
_output_shapes
::���
%model/user_encoder/att_layer2/unstackUnpack,model/user_encoder/att_layer2/Shape:output:0*
T0*
_output_shapes
: : : *	
num�
4model/user_encoder/att_layer2/Shape_1/ReadVariableOpReadVariableOp=model_user_encoder_att_layer2_shape_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0v
%model/user_encoder/att_layer2/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"�  �   �
'model/user_encoder/att_layer2/unstack_1Unpack.model/user_encoder/att_layer2/Shape_1:output:0*
T0*
_output_shapes
: : *	
num|
+model/user_encoder/att_layer2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
%model/user_encoder/att_layer2/ReshapeReshape5model/user_encoder/self_attention/Reshape_12:output:04model/user_encoder/att_layer2/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
6model/user_encoder/att_layer2/transpose/ReadVariableOpReadVariableOp=model_user_encoder_att_layer2_shape_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0}
,model/user_encoder/att_layer2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
'model/user_encoder/att_layer2/transpose	Transpose>model/user_encoder/att_layer2/transpose/ReadVariableOp:value:05model/user_encoder/att_layer2/transpose/perm:output:0*
T0* 
_output_shapes
:
��~
-model/user_encoder/att_layer2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"�  �����
'model/user_encoder/att_layer2/Reshape_1Reshape+model/user_encoder/att_layer2/transpose:y:06model/user_encoder/att_layer2/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
$model/user_encoder/att_layer2/MatMulMatMul.model/user_encoder/att_layer2/Reshape:output:00model/user_encoder/att_layer2/Reshape_1:output:0*
T0*(
_output_shapes
:����������q
/model/user_encoder/att_layer2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#r
/model/user_encoder/att_layer2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
-model/user_encoder/att_layer2/Reshape_2/shapePack.model/user_encoder/att_layer2/unstack:output:08model/user_encoder/att_layer2/Reshape_2/shape/1:output:08model/user_encoder/att_layer2/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:�
'model/user_encoder/att_layer2/Reshape_2Reshape.model/user_encoder/att_layer2/MatMul:product:06model/user_encoder/att_layer2/Reshape_2/shape:output:0*
T0*,
_output_shapes
:���������#��
0model/user_encoder/att_layer2/add/ReadVariableOpReadVariableOp9model_user_encoder_att_layer2_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!model/user_encoder/att_layer2/addAddV20model/user_encoder/att_layer2/Reshape_2:output:08model/user_encoder/att_layer2/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������#��
"model/user_encoder/att_layer2/TanhTanh%model/user_encoder/att_layer2/add:z:0*
T0*,
_output_shapes
:���������#��
%model/user_encoder/att_layer2/Shape_2Shape&model/user_encoder/att_layer2/Tanh:y:0*
T0*
_output_shapes
::���
'model/user_encoder/att_layer2/unstack_2Unpack.model/user_encoder/att_layer2/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num�
4model/user_encoder/att_layer2/Shape_3/ReadVariableOpReadVariableOp=model_user_encoder_att_layer2_shape_3_readvariableop_resource*
_output_shapes
:	�*
dtype0v
%model/user_encoder/att_layer2/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"�      �
'model/user_encoder/att_layer2/unstack_3Unpack.model/user_encoder/att_layer2/Shape_3:output:0*
T0*
_output_shapes
: : *	
num~
-model/user_encoder/att_layer2/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
'model/user_encoder/att_layer2/Reshape_3Reshape&model/user_encoder/att_layer2/Tanh:y:06model/user_encoder/att_layer2/Reshape_3/shape:output:0*
T0*(
_output_shapes
:�����������
8model/user_encoder/att_layer2/transpose_1/ReadVariableOpReadVariableOp=model_user_encoder_att_layer2_shape_3_readvariableop_resource*
_output_shapes
:	�*
dtype0
.model/user_encoder/att_layer2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       �
)model/user_encoder/att_layer2/transpose_1	Transpose@model/user_encoder/att_layer2/transpose_1/ReadVariableOp:value:07model/user_encoder/att_layer2/transpose_1/perm:output:0*
T0*
_output_shapes
:	�~
-model/user_encoder/att_layer2/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"�   �����
'model/user_encoder/att_layer2/Reshape_4Reshape-model/user_encoder/att_layer2/transpose_1:y:06model/user_encoder/att_layer2/Reshape_4/shape:output:0*
T0*
_output_shapes
:	��
&model/user_encoder/att_layer2/MatMul_1MatMul0model/user_encoder/att_layer2/Reshape_3:output:00model/user_encoder/att_layer2/Reshape_4:output:0*
T0*'
_output_shapes
:���������q
/model/user_encoder/att_layer2/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#q
/model/user_encoder/att_layer2/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
-model/user_encoder/att_layer2/Reshape_5/shapePack0model/user_encoder/att_layer2/unstack_2:output:08model/user_encoder/att_layer2/Reshape_5/shape/1:output:08model/user_encoder/att_layer2/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:�
'model/user_encoder/att_layer2/Reshape_5Reshape0model/user_encoder/att_layer2/MatMul_1:product:06model/user_encoder/att_layer2/Reshape_5/shape:output:0*
T0*+
_output_shapes
:���������#�
%model/user_encoder/att_layer2/SqueezeSqueeze0model/user_encoder/att_layer2/Reshape_5:output:0*
T0*'
_output_shapes
:���������#*
squeeze_dims
�
!model/user_encoder/att_layer2/ExpExp.model/user_encoder/att_layer2/Squeeze:output:0*
T0*'
_output_shapes
:���������#~
3model/user_encoder/att_layer2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
!model/user_encoder/att_layer2/SumSum%model/user_encoder/att_layer2/Exp:y:0<model/user_encoder/att_layer2/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(j
%model/user_encoder/att_layer2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
#model/user_encoder/att_layer2/add_1AddV2*model/user_encoder/att_layer2/Sum:output:0.model/user_encoder/att_layer2/add_1/y:output:0*
T0*'
_output_shapes
:����������
%model/user_encoder/att_layer2/truedivRealDiv%model/user_encoder/att_layer2/Exp:y:0'model/user_encoder/att_layer2/add_1:z:0*
T0*'
_output_shapes
:���������#w
,model/user_encoder/att_layer2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
(model/user_encoder/att_layer2/ExpandDims
ExpandDims)model/user_encoder/att_layer2/truediv:z:05model/user_encoder/att_layer2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������#�
!model/user_encoder/att_layer2/mulMul5model/user_encoder/self_attention/Reshape_12:output:01model/user_encoder/att_layer2/ExpandDims:output:0*
T0*,
_output_shapes
:���������#�w
5model/user_encoder/att_layer2/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
#model/user_encoder/att_layer2/Sum_1Sum%model/user_encoder/att_layer2/mul:z:0>model/user_encoder/att_layer2/Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:����������Z
model/dot/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/dot/ExpandDims
ExpandDims,model/user_encoder/att_layer2/Sum_1:output:0!model/dot/ExpandDims/dim:output:0*
T0*,
_output_shapes
:�����������
model/dot/MatMulBatchMatMulV2+model/time_distributed_1/Reshape_1:output:0model/dot/ExpandDims:output:0*
T0*4
_output_shapes"
 :������������������f
model/dot/ShapeShapemodel/dot/MatMul:output:0*
T0*
_output_shapes
::���
model/dot/SqueezeSqueezemodel/dot/MatMul:output:0*
T0*0
_output_shapes
:������������������*
squeeze_dims

���������z
model/activation/SoftmaxSoftmaxmodel/dot/Squeeze:output:0*
T0*0
_output_shapes
:������������������z
IdentityIdentity"model/activation/Softmax:softmax:0^NoOp*
T0*0
_output_shapes
:�������������������5
NoOpNoOpF^model/time_distributed_1/batch_normalization/batchnorm/ReadVariableOpH^model/time_distributed_1/batch_normalization/batchnorm/ReadVariableOp_1H^model/time_distributed_1/batch_normalization/batchnorm/ReadVariableOp_2J^model/time_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOpH^model/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOpJ^model/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_1J^model/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_2L^model/time_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOpH^model/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOpJ^model/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_1J^model/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_2L^model/time_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOp6^model/time_distributed_1/dense/BiasAdd/ReadVariableOp5^model/time_distributed_1/dense/MatMul/ReadVariableOp8^model/time_distributed_1/dense_1/BiasAdd/ReadVariableOp7^model/time_distributed_1/dense_1/MatMul/ReadVariableOp8^model/time_distributed_1/dense_2/BiasAdd/ReadVariableOp7^model/time_distributed_1/dense_2/MatMul/ReadVariableOp8^model/time_distributed_1/dense_3/BiasAdd/ReadVariableOp7^model/time_distributed_1/dense_3/MatMul/ReadVariableOpS^model/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOpU^model/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOp_1U^model/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOp_2W^model/time_distributed_1/news_encoder/batch_normalization/batchnorm/mul/ReadVariableOpU^model/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOpW^model/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1W^model/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2Y^model/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOpU^model/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOpW^model/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1W^model/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2Y^model/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOpC^model/time_distributed_1/news_encoder/dense/BiasAdd/ReadVariableOpB^model/time_distributed_1/news_encoder/dense/MatMul/ReadVariableOpE^model/time_distributed_1/news_encoder/dense_1/BiasAdd/ReadVariableOpD^model/time_distributed_1/news_encoder/dense_1/MatMul/ReadVariableOpE^model/time_distributed_1/news_encoder/dense_2/BiasAdd/ReadVariableOpD^model/time_distributed_1/news_encoder/dense_2/MatMul/ReadVariableOpE^model/time_distributed_1/news_encoder/dense_3/BiasAdd/ReadVariableOpD^model/time_distributed_1/news_encoder/dense_3/MatMul/ReadVariableOp1^model/user_encoder/att_layer2/add/ReadVariableOp7^model/user_encoder/att_layer2/transpose/ReadVariableOp9^model/user_encoder/att_layer2/transpose_1/ReadVariableOp;^model/user_encoder/self_attention/transpose/ReadVariableOp=^model/user_encoder/self_attention/transpose_2/ReadVariableOp=^model/user_encoder/self_attention/transpose_4/ReadVariableOpQ^model/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOpS^model/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOp_1S^model/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOp_2U^model/user_encoder/time_distributed/batch_normalization/batchnorm/mul/ReadVariableOpS^model/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOpU^model/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOp_1U^model/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOp_2W^model/user_encoder/time_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOpS^model/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOpU^model/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOp_1U^model/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOp_2W^model/user_encoder/time_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOpA^model/user_encoder/time_distributed/dense/BiasAdd/ReadVariableOp@^model/user_encoder/time_distributed/dense/MatMul/ReadVariableOpC^model/user_encoder/time_distributed/dense_1/BiasAdd/ReadVariableOpB^model/user_encoder/time_distributed/dense_1/MatMul/ReadVariableOpC^model/user_encoder/time_distributed/dense_2/BiasAdd/ReadVariableOpB^model/user_encoder/time_distributed/dense_2/MatMul/ReadVariableOpC^model/user_encoder/time_distributed/dense_3/BiasAdd/ReadVariableOpB^model/user_encoder/time_distributed/dense_3/MatMul/ReadVariableOp^^model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp`^model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp_1`^model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp_2b^model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp`^model/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOpb^model/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1b^model/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2d^model/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp`^model/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOpb^model/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1b^model/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2d^model/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOpN^model/user_encoder/time_distributed/news_encoder/dense/BiasAdd/ReadVariableOpM^model/user_encoder/time_distributed/news_encoder/dense/MatMul/ReadVariableOpP^model/user_encoder/time_distributed/news_encoder/dense_1/BiasAdd/ReadVariableOpO^model/user_encoder/time_distributed/news_encoder/dense_1/MatMul/ReadVariableOpP^model/user_encoder/time_distributed/news_encoder/dense_2/BiasAdd/ReadVariableOpO^model/user_encoder/time_distributed/news_encoder/dense_2/MatMul/ReadVariableOpP^model/user_encoder/time_distributed/news_encoder/dense_3/BiasAdd/ReadVariableOpO^model/user_encoder/time_distributed/news_encoder/dense_3/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:���������#�:�������������������: : : : : : : : : : : : : : : : : : : : : : : : : : 2�
Gmodel/time_distributed_1/batch_normalization/batchnorm/ReadVariableOp_1Gmodel/time_distributed_1/batch_normalization/batchnorm/ReadVariableOp_12�
Gmodel/time_distributed_1/batch_normalization/batchnorm/ReadVariableOp_2Gmodel/time_distributed_1/batch_normalization/batchnorm/ReadVariableOp_22�
Emodel/time_distributed_1/batch_normalization/batchnorm/ReadVariableOpEmodel/time_distributed_1/batch_normalization/batchnorm/ReadVariableOp2�
Imodel/time_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOpImodel/time_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOp2�
Imodel/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_1Imodel/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_12�
Imodel/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_2Imodel/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_22�
Gmodel/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOpGmodel/time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp2�
Kmodel/time_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOpKmodel/time_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOp2�
Imodel/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_1Imodel/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_12�
Imodel/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_2Imodel/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_22�
Gmodel/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOpGmodel/time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp2�
Kmodel/time_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOpKmodel/time_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOp2n
5model/time_distributed_1/dense/BiasAdd/ReadVariableOp5model/time_distributed_1/dense/BiasAdd/ReadVariableOp2l
4model/time_distributed_1/dense/MatMul/ReadVariableOp4model/time_distributed_1/dense/MatMul/ReadVariableOp2r
7model/time_distributed_1/dense_1/BiasAdd/ReadVariableOp7model/time_distributed_1/dense_1/BiasAdd/ReadVariableOp2p
6model/time_distributed_1/dense_1/MatMul/ReadVariableOp6model/time_distributed_1/dense_1/MatMul/ReadVariableOp2r
7model/time_distributed_1/dense_2/BiasAdd/ReadVariableOp7model/time_distributed_1/dense_2/BiasAdd/ReadVariableOp2p
6model/time_distributed_1/dense_2/MatMul/ReadVariableOp6model/time_distributed_1/dense_2/MatMul/ReadVariableOp2r
7model/time_distributed_1/dense_3/BiasAdd/ReadVariableOp7model/time_distributed_1/dense_3/BiasAdd/ReadVariableOp2p
6model/time_distributed_1/dense_3/MatMul/ReadVariableOp6model/time_distributed_1/dense_3/MatMul/ReadVariableOp2�
Tmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOp_1Tmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOp_12�
Tmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOp_2Tmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOp_22�
Rmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOpRmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/ReadVariableOp2�
Vmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/mul/ReadVariableOpVmodel/time_distributed_1/news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp2�
Vmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1Vmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_12�
Vmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2Vmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_22�
Tmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOpTmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp2�
Xmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOpXmodel/time_distributed_1/news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp2�
Vmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1Vmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_12�
Vmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2Vmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_22�
Tmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOpTmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp2�
Xmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOpXmodel/time_distributed_1/news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp2�
Bmodel/time_distributed_1/news_encoder/dense/BiasAdd/ReadVariableOpBmodel/time_distributed_1/news_encoder/dense/BiasAdd/ReadVariableOp2�
Amodel/time_distributed_1/news_encoder/dense/MatMul/ReadVariableOpAmodel/time_distributed_1/news_encoder/dense/MatMul/ReadVariableOp2�
Dmodel/time_distributed_1/news_encoder/dense_1/BiasAdd/ReadVariableOpDmodel/time_distributed_1/news_encoder/dense_1/BiasAdd/ReadVariableOp2�
Cmodel/time_distributed_1/news_encoder/dense_1/MatMul/ReadVariableOpCmodel/time_distributed_1/news_encoder/dense_1/MatMul/ReadVariableOp2�
Dmodel/time_distributed_1/news_encoder/dense_2/BiasAdd/ReadVariableOpDmodel/time_distributed_1/news_encoder/dense_2/BiasAdd/ReadVariableOp2�
Cmodel/time_distributed_1/news_encoder/dense_2/MatMul/ReadVariableOpCmodel/time_distributed_1/news_encoder/dense_2/MatMul/ReadVariableOp2�
Dmodel/time_distributed_1/news_encoder/dense_3/BiasAdd/ReadVariableOpDmodel/time_distributed_1/news_encoder/dense_3/BiasAdd/ReadVariableOp2�
Cmodel/time_distributed_1/news_encoder/dense_3/MatMul/ReadVariableOpCmodel/time_distributed_1/news_encoder/dense_3/MatMul/ReadVariableOp2d
0model/user_encoder/att_layer2/add/ReadVariableOp0model/user_encoder/att_layer2/add/ReadVariableOp2p
6model/user_encoder/att_layer2/transpose/ReadVariableOp6model/user_encoder/att_layer2/transpose/ReadVariableOp2t
8model/user_encoder/att_layer2/transpose_1/ReadVariableOp8model/user_encoder/att_layer2/transpose_1/ReadVariableOp2x
:model/user_encoder/self_attention/transpose/ReadVariableOp:model/user_encoder/self_attention/transpose/ReadVariableOp2|
<model/user_encoder/self_attention/transpose_2/ReadVariableOp<model/user_encoder/self_attention/transpose_2/ReadVariableOp2|
<model/user_encoder/self_attention/transpose_4/ReadVariableOp<model/user_encoder/self_attention/transpose_4/ReadVariableOp2�
Rmodel/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOp_1Rmodel/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOp_12�
Rmodel/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOp_2Rmodel/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOp_22�
Pmodel/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOpPmodel/user_encoder/time_distributed/batch_normalization/batchnorm/ReadVariableOp2�
Tmodel/user_encoder/time_distributed/batch_normalization/batchnorm/mul/ReadVariableOpTmodel/user_encoder/time_distributed/batch_normalization/batchnorm/mul/ReadVariableOp2�
Tmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOp_1Tmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOp_12�
Tmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOp_2Tmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOp_22�
Rmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOpRmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/ReadVariableOp2�
Vmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOpVmodel/user_encoder/time_distributed/batch_normalization_1/batchnorm/mul/ReadVariableOp2�
Tmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOp_1Tmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOp_12�
Tmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOp_2Tmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOp_22�
Rmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOpRmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/ReadVariableOp2�
Vmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOpVmodel/user_encoder/time_distributed/batch_normalization_2/batchnorm/mul/ReadVariableOp2�
@model/user_encoder/time_distributed/dense/BiasAdd/ReadVariableOp@model/user_encoder/time_distributed/dense/BiasAdd/ReadVariableOp2�
?model/user_encoder/time_distributed/dense/MatMul/ReadVariableOp?model/user_encoder/time_distributed/dense/MatMul/ReadVariableOp2�
Bmodel/user_encoder/time_distributed/dense_1/BiasAdd/ReadVariableOpBmodel/user_encoder/time_distributed/dense_1/BiasAdd/ReadVariableOp2�
Amodel/user_encoder/time_distributed/dense_1/MatMul/ReadVariableOpAmodel/user_encoder/time_distributed/dense_1/MatMul/ReadVariableOp2�
Bmodel/user_encoder/time_distributed/dense_2/BiasAdd/ReadVariableOpBmodel/user_encoder/time_distributed/dense_2/BiasAdd/ReadVariableOp2�
Amodel/user_encoder/time_distributed/dense_2/MatMul/ReadVariableOpAmodel/user_encoder/time_distributed/dense_2/MatMul/ReadVariableOp2�
Bmodel/user_encoder/time_distributed/dense_3/BiasAdd/ReadVariableOpBmodel/user_encoder/time_distributed/dense_3/BiasAdd/ReadVariableOp2�
Amodel/user_encoder/time_distributed/dense_3/MatMul/ReadVariableOpAmodel/user_encoder/time_distributed/dense_3/MatMul/ReadVariableOp2�
_model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp_1_model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp_12�
_model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp_2_model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp_22�
]model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp]model/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/ReadVariableOp2�
amodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/mul/ReadVariableOpamodel/user_encoder/time_distributed/news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp2�
amodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_1amodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_12�
amodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_2amodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_22�
_model/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp_model/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/ReadVariableOp2�
cmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOpcmodel/user_encoder/time_distributed/news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp2�
amodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_1amodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_12�
amodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_2amodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_22�
_model/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp_model/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/ReadVariableOp2�
cmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOpcmodel/user_encoder/time_distributed/news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp2�
Mmodel/user_encoder/time_distributed/news_encoder/dense/BiasAdd/ReadVariableOpMmodel/user_encoder/time_distributed/news_encoder/dense/BiasAdd/ReadVariableOp2�
Lmodel/user_encoder/time_distributed/news_encoder/dense/MatMul/ReadVariableOpLmodel/user_encoder/time_distributed/news_encoder/dense/MatMul/ReadVariableOp2�
Omodel/user_encoder/time_distributed/news_encoder/dense_1/BiasAdd/ReadVariableOpOmodel/user_encoder/time_distributed/news_encoder/dense_1/BiasAdd/ReadVariableOp2�
Nmodel/user_encoder/time_distributed/news_encoder/dense_1/MatMul/ReadVariableOpNmodel/user_encoder/time_distributed/news_encoder/dense_1/MatMul/ReadVariableOp2�
Omodel/user_encoder/time_distributed/news_encoder/dense_2/BiasAdd/ReadVariableOpOmodel/user_encoder/time_distributed/news_encoder/dense_2/BiasAdd/ReadVariableOp2�
Nmodel/user_encoder/time_distributed/news_encoder/dense_2/MatMul/ReadVariableOpNmodel/user_encoder/time_distributed/news_encoder/dense_2/MatMul/ReadVariableOp2�
Omodel/user_encoder/time_distributed/news_encoder/dense_3/BiasAdd/ReadVariableOpOmodel/user_encoder/time_distributed/news_encoder/dense_3/BiasAdd/ReadVariableOp2�
Nmodel/user_encoder/time_distributed/news_encoder/dense_3/MatMul/ReadVariableOpNmodel/user_encoder/time_distributed/news_encoder/dense_3/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^Z
5
_output_shapes#
!:�������������������
!
_user_specified_name	input_2:U Q
,
_output_shapes
:���������#�
!
_user_specified_name	input_1
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_83082

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_dense_1_layer_call_fn_86338

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_83239p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name86334:%!

_user_specified_name86332:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
E
)__inference_dropout_1_layer_call_fn_86439

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_83363a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
B__inference_dropout_layer_call_and_return_conditional_losses_83343

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

a
B__inference_dropout_layer_call_and_return_conditional_losses_83227

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_83265

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�i
�
I__inference_self_attention_layer_call_and_return_conditional_losses_86056

qkvs_0

qkvs_1

qkvs_23
shape_1_readvariableop_resource:
��3
shape_4_readvariableop_resource:
��3
shape_7_readvariableop_resource:
��
identity��transpose/ReadVariableOp�transpose_2/ReadVariableOp�transpose_4/ReadVariableOpI
ShapeShapeqkvs_0*
T0*
_output_shapes
::��Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"�  �  S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  e
ReshapeReshapeqkvs_0Reshape/shape:output:0*
T0*(
_output_shapes
:����������z
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       |
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0* 
_output_shapes
:
��`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"�  ����h
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
��i
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*(
_output_shapes
:����������S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#T
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:w
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*,
_output_shapes
:���������#�W
Shape_2ShapeReshape_2:output:0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape_2:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_3/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_3/shapePackReshape_3/shape/0:output:0strided_slice:output:0Reshape_3/shape/2:output:0Reshape_3/shape/3:output:0*
N*
T0*
_output_shapes
:|
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*/
_output_shapes
:���������#i
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
transpose_1	TransposeReshape_3:output:0transpose_1/perm:output:0*
T0*/
_output_shapes
:���������#K
Shape_3Shapeqkvs_1*
T0*
_output_shapes
::��U
	unstack_2UnpackShape_3:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_4/ReadVariableOpReadVariableOpshape_4_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Shape_4Const*
_output_shapes
:*
dtype0*
valueB"�  �  S
	unstack_3UnpackShape_4:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  i
	Reshape_4Reshapeqkvs_1Reshape_4/shape:output:0*
T0*(
_output_shapes
:����������|
transpose_2/ReadVariableOpReadVariableOpshape_4_readvariableop_resource* 
_output_shapes
:
��*
dtype0a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       �
transpose_2	Transpose"transpose_2/ReadVariableOp:value:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
��`
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB"�  ����j
	Reshape_5Reshapetranspose_2:y:0Reshape_5/shape:output:0*
T0* 
_output_shapes
:
��m
MatMul_1MatMulReshape_4:output:0Reshape_5:output:0*
T0*(
_output_shapes
:����������S
Reshape_6/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#T
Reshape_6/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_6/shapePackunstack_2:output:0Reshape_6/shape/1:output:0Reshape_6/shape/2:output:0*
N*
T0*
_output_shapes
:y
	Reshape_6ReshapeMatMul_1:product:0Reshape_6/shape:output:0*
T0*,
_output_shapes
:���������#�W
Shape_5ShapeReshape_6:output:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_5:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
Reshape_7/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_7/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_7/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_7/shapePackReshape_7/shape/0:output:0strided_slice_1:output:0Reshape_7/shape/2:output:0Reshape_7/shape/3:output:0*
N*
T0*
_output_shapes
:|
	Reshape_7ReshapeReshape_6:output:0Reshape_7/shape:output:0*
T0*/
_output_shapes
:���������#i
transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
transpose_3	TransposeReshape_7:output:0transpose_3/perm:output:0*
T0*/
_output_shapes
:���������#K
Shape_6Shapeqkvs_2*
T0*
_output_shapes
::��U
	unstack_4UnpackShape_6:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_7/ReadVariableOpReadVariableOpshape_7_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Shape_7Const*
_output_shapes
:*
dtype0*
valueB"�  �  S
	unstack_5UnpackShape_7:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  i
	Reshape_8Reshapeqkvs_2Reshape_8/shape:output:0*
T0*(
_output_shapes
:����������|
transpose_4/ReadVariableOpReadVariableOpshape_7_readvariableop_resource* 
_output_shapes
:
��*
dtype0a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       �
transpose_4	Transpose"transpose_4/ReadVariableOp:value:0transpose_4/perm:output:0*
T0* 
_output_shapes
:
��`
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"�  ����j
	Reshape_9Reshapetranspose_4:y:0Reshape_9/shape:output:0*
T0* 
_output_shapes
:
��m
MatMul_2MatMulReshape_8:output:0Reshape_9:output:0*
T0*(
_output_shapes
:����������T
Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#U
Reshape_10/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_10/shapePackunstack_4:output:0Reshape_10/shape/1:output:0Reshape_10/shape/2:output:0*
N*
T0*
_output_shapes
:{

Reshape_10ReshapeMatMul_2:product:0Reshape_10/shape:output:0*
T0*,
_output_shapes
:���������#�X
Shape_8ShapeReshape_10:output:0*
T0*
_output_shapes
::��_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_8:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
Reshape_11/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_11/shape/2Const*
_output_shapes
: *
dtype0*
value	B :T
Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_11/shapePackReshape_11/shape/0:output:0strided_slice_2:output:0Reshape_11/shape/2:output:0Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_11ReshapeReshape_10:output:0Reshape_11/shape:output:0*
T0*/
_output_shapes
:���������#i
transpose_5/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
transpose_5	TransposeReshape_11:output:0transpose_5/perm:output:0*
T0*/
_output_shapes
:���������#�
MatMul_3BatchMatMulV2transpose_1:y:0transpose_3:y:0*
T0*/
_output_shapes
:���������##*
adj_y(H
Cast/xConst*
_output_shapes
: *
dtype0*
value	B :M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    M
MaximumMaximumCast:y:0Const:output:0*
T0*
_output_shapes
: :
SqrtSqrtMaximum:z:0*
T0*
_output_shapes
: i
truedivRealDivMatMul_3:output:0Sqrt:y:0*
T0*/
_output_shapes
:���������##i
transpose_6/permConst*
_output_shapes
:*
dtype0*%
valueB"             z
transpose_6	Transposetruediv:z:0transpose_6/perm:output:0*
T0*/
_output_shapes
:���������##i
transpose_7/permConst*
_output_shapes
:*
dtype0*%
valueB"             ~
transpose_7	Transposetranspose_6:y:0transpose_7/perm:output:0*
T0*/
_output_shapes
:���������##]
SoftmaxSoftmaxtranspose_7:y:0*
T0*/
_output_shapes
:���������##�
MatMul_4BatchMatMulV2Softmax:softmax:0transpose_5:y:0*
T0*/
_output_shapes
:���������#*
adj_x(i
transpose_8/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
transpose_8	TransposeMatMul_4:output:0transpose_8/perm:output:0*
T0*/
_output_shapes
:���������#T
Shape_9Shapetranspose_8:y:0*
T0*
_output_shapes
::��_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_9:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
Reshape_12/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������U
Reshape_12/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_12/shapePackReshape_12/shape/0:output:0strided_slice_3:output:0Reshape_12/shape/2:output:0*
N*
T0*
_output_shapes
:x

Reshape_12Reshapetranspose_8:y:0Reshape_12/shape:output:0*
T0*,
_output_shapes
:���������#�g
IdentityIdentityReshape_12:output:0^NoOp*
T0*,
_output_shapes
:���������#�w
NoOpNoOp^transpose/ReadVariableOp^transpose_2/ReadVariableOp^transpose_4/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:���������#�:���������#�:���������#�: : : 24
transpose/ReadVariableOptranspose/ReadVariableOp28
transpose_2/ReadVariableOptranspose_2/ReadVariableOp28
transpose_4/ReadVariableOptranspose_4/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:TP
,
_output_shapes
:���������#�
 
_user_specified_nameqkvs_2:TP
,
_output_shapes
:���������#�
 
_user_specified_nameqkvs_1:T P
,
_output_shapes
:���������#�
 
_user_specified_nameqkvs_0
�	
�
5__inference_batch_normalization_1_layer_call_fn_86362

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_83062p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name86358:%!

_user_specified_name86356:%!

_user_specified_name86354:%!

_user_specified_name86352:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�!
�
K__inference_time_distributed_layer_call_and_return_conditional_losses_83805

inputs&
news_encoder_83759:
��!
news_encoder_83761:	�!
news_encoder_83763:	�!
news_encoder_83765:	�!
news_encoder_83767:	�!
news_encoder_83769:	�&
news_encoder_83771:
��!
news_encoder_83773:	�!
news_encoder_83775:	�!
news_encoder_83777:	�!
news_encoder_83779:	�!
news_encoder_83781:	�&
news_encoder_83783:
��!
news_encoder_83785:	�!
news_encoder_83787:	�!
news_encoder_83789:	�!
news_encoder_83791:	�!
news_encoder_83793:	�&
news_encoder_83795:
��!
news_encoder_83797:	�
identity��$news_encoder/StatefulPartitionedCallI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
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
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:�����������
$news_encoder/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0news_encoder_83759news_encoder_83761news_encoder_83763news_encoder_83765news_encoder_83767news_encoder_83769news_encoder_83771news_encoder_83773news_encoder_83775news_encoder_83777news_encoder_83779news_encoder_83781news_encoder_83783news_encoder_83785news_encoder_83787news_encoder_83789news_encoder_83791news_encoder_83793news_encoder_83795news_encoder_83797* 
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*0
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *P
fKRI
G__inference_news_encoder_layer_call_and_return_conditional_losses_83322\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape-news_encoder/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������I
NoOpNoOp%^news_encoder/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�������������������: : : : : : : : : : : : : : : : : : : : 2L
$news_encoder/StatefulPartitionedCall$news_encoder/StatefulPartitionedCall:%!

_user_specified_name83797:%!

_user_specified_name83795:%!

_user_specified_name83793:%!

_user_specified_name83791:%!

_user_specified_name83789:%!

_user_specified_name83787:%!

_user_specified_name83785:%!

_user_specified_name83783:%!

_user_specified_name83781:%!

_user_specified_name83779:%
!

_user_specified_name83777:%	!

_user_specified_name83775:%!

_user_specified_name83773:%!

_user_specified_name83771:%!

_user_specified_name83769:%!

_user_specified_name83767:%!

_user_specified_name83765:%!

_user_specified_name83763:%!

_user_specified_name83761:%!

_user_specified_name83759:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
@__inference_model_layer_call_and_return_conditional_losses_85018
input_1
input_2,
time_distributed_1_84880:
��'
time_distributed_1_84882:	�'
time_distributed_1_84884:	�'
time_distributed_1_84886:	�'
time_distributed_1_84888:	�'
time_distributed_1_84890:	�,
time_distributed_1_84892:
��'
time_distributed_1_84894:	�'
time_distributed_1_84896:	�'
time_distributed_1_84898:	�'
time_distributed_1_84900:	�'
time_distributed_1_84902:	�,
time_distributed_1_84904:
��'
time_distributed_1_84906:	�'
time_distributed_1_84908:	�'
time_distributed_1_84910:	�'
time_distributed_1_84912:	�'
time_distributed_1_84914:	�,
time_distributed_1_84916:
��'
time_distributed_1_84918:	�&
user_encoder_85002:
��&
user_encoder_85004:
��&
user_encoder_85006:
��&
user_encoder_85008:
��!
user_encoder_85010:	�%
user_encoder_85012:	�
identity��*time_distributed_1/StatefulPartitionedCall�?time_distributed_1/batch_normalization/batchnorm/ReadVariableOp�Atime_distributed_1/batch_normalization/batchnorm/ReadVariableOp_1�Atime_distributed_1/batch_normalization/batchnorm/ReadVariableOp_2�Ctime_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOp�Atime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp�Ctime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_1�Ctime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_2�Etime_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOp�Atime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp�Ctime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_1�Ctime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_2�Etime_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOp�/time_distributed_1/dense/BiasAdd/ReadVariableOp�.time_distributed_1/dense/MatMul/ReadVariableOp�1time_distributed_1/dense_1/BiasAdd/ReadVariableOp�0time_distributed_1/dense_1/MatMul/ReadVariableOp�1time_distributed_1/dense_2/BiasAdd/ReadVariableOp�0time_distributed_1/dense_2/MatMul/ReadVariableOp�1time_distributed_1/dense_3/BiasAdd/ReadVariableOp�0time_distributed_1/dense_3/MatMul/ReadVariableOp�$user_encoder/StatefulPartitionedCall�
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCallinput_2time_distributed_1_84880time_distributed_1_84882time_distributed_1_84884time_distributed_1_84886time_distributed_1_84888time_distributed_1_84890time_distributed_1_84892time_distributed_1_84894time_distributed_1_84896time_distributed_1_84898time_distributed_1_84900time_distributed_1_84902time_distributed_1_84904time_distributed_1_84906time_distributed_1_84908time_distributed_1_84910time_distributed_1_84912time_distributed_1_84914time_distributed_1_84916time_distributed_1_84918* 
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*6
_read_only_resource_inputs
	
*<
config_proto,*

CPU

GPU 2J 8R(���������� *V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_83659q
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
time_distributed_1/ReshapeReshapeinput_2)time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
.time_distributed_1/dense/MatMul/ReadVariableOpReadVariableOptime_distributed_1_84880* 
_output_shapes
:
��*
dtype0�
time_distributed_1/dense/MatMulMatMul#time_distributed_1/Reshape:output:06time_distributed_1/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/time_distributed_1/dense/BiasAdd/ReadVariableOpReadVariableOptime_distributed_1_84882*
_output_shapes	
:�*
dtype0�
 time_distributed_1/dense/BiasAddBiasAdd)time_distributed_1/dense/MatMul:product:07time_distributed_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
time_distributed_1/dense/ReluRelu)time_distributed_1/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
?time_distributed_1/batch_normalization/batchnorm/ReadVariableOpReadVariableOptime_distributed_1_84884*
_output_shapes	
:�*
dtype0{
6time_distributed_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4time_distributed_1/batch_normalization/batchnorm/addAddV2Gtime_distributed_1/batch_normalization/batchnorm/ReadVariableOp:value:0?time_distributed_1/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
6time_distributed_1/batch_normalization/batchnorm/RsqrtRsqrt8time_distributed_1/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Ctime_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOptime_distributed_1_84886*
_output_shapes	
:�*
dtype0�
4time_distributed_1/batch_normalization/batchnorm/mulMul:time_distributed_1/batch_normalization/batchnorm/Rsqrt:y:0Ktime_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
6time_distributed_1/batch_normalization/batchnorm/mul_1Mul+time_distributed_1/dense/Relu:activations:08time_distributed_1/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Atime_distributed_1/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOptime_distributed_1_84888*
_output_shapes	
:�*
dtype0�
6time_distributed_1/batch_normalization/batchnorm/mul_2MulItime_distributed_1/batch_normalization/batchnorm/ReadVariableOp_1:value:08time_distributed_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
Atime_distributed_1/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOptime_distributed_1_84890*
_output_shapes	
:�*
dtype0�
4time_distributed_1/batch_normalization/batchnorm/subSubItime_distributed_1/batch_normalization/batchnorm/ReadVariableOp_2:value:0:time_distributed_1/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
6time_distributed_1/batch_normalization/batchnorm/add_1AddV2:time_distributed_1/batch_normalization/batchnorm/mul_1:z:08time_distributed_1/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
#time_distributed_1/dropout/IdentityIdentity:time_distributed_1/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
0time_distributed_1/dense_1/MatMul/ReadVariableOpReadVariableOptime_distributed_1_84892* 
_output_shapes
:
��*
dtype0�
!time_distributed_1/dense_1/MatMulMatMul,time_distributed_1/dropout/Identity:output:08time_distributed_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
1time_distributed_1/dense_1/BiasAdd/ReadVariableOpReadVariableOptime_distributed_1_84894*
_output_shapes	
:�*
dtype0�
"time_distributed_1/dense_1/BiasAddBiasAdd+time_distributed_1/dense_1/MatMul:product:09time_distributed_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
time_distributed_1/dense_1/ReluRelu+time_distributed_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Atime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOptime_distributed_1_84896*
_output_shapes	
:�*
dtype0}
8time_distributed_1/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
6time_distributed_1/batch_normalization_1/batchnorm/addAddV2Itime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp:value:0Atime_distributed_1/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
8time_distributed_1/batch_normalization_1/batchnorm/RsqrtRsqrt:time_distributed_1/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Etime_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOptime_distributed_1_84898*
_output_shapes	
:�*
dtype0�
6time_distributed_1/batch_normalization_1/batchnorm/mulMul<time_distributed_1/batch_normalization_1/batchnorm/Rsqrt:y:0Mtime_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
8time_distributed_1/batch_normalization_1/batchnorm/mul_1Mul-time_distributed_1/dense_1/Relu:activations:0:time_distributed_1/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Ctime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOptime_distributed_1_84900*
_output_shapes	
:�*
dtype0�
8time_distributed_1/batch_normalization_1/batchnorm/mul_2MulKtime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0:time_distributed_1/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
Ctime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOptime_distributed_1_84902*
_output_shapes	
:�*
dtype0�
6time_distributed_1/batch_normalization_1/batchnorm/subSubKtime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0<time_distributed_1/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
8time_distributed_1/batch_normalization_1/batchnorm/add_1AddV2<time_distributed_1/batch_normalization_1/batchnorm/mul_1:z:0:time_distributed_1/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
%time_distributed_1/dropout_1/IdentityIdentity<time_distributed_1/batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
0time_distributed_1/dense_2/MatMul/ReadVariableOpReadVariableOptime_distributed_1_84904* 
_output_shapes
:
��*
dtype0�
!time_distributed_1/dense_2/MatMulMatMul.time_distributed_1/dropout_1/Identity:output:08time_distributed_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
1time_distributed_1/dense_2/BiasAdd/ReadVariableOpReadVariableOptime_distributed_1_84906*
_output_shapes	
:�*
dtype0�
"time_distributed_1/dense_2/BiasAddBiasAdd+time_distributed_1/dense_2/MatMul:product:09time_distributed_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
time_distributed_1/dense_2/ReluRelu+time_distributed_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Atime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOptime_distributed_1_84908*
_output_shapes	
:�*
dtype0}
8time_distributed_1/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
6time_distributed_1/batch_normalization_2/batchnorm/addAddV2Itime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp:value:0Atime_distributed_1/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
8time_distributed_1/batch_normalization_2/batchnorm/RsqrtRsqrt:time_distributed_1/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Etime_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOptime_distributed_1_84910*
_output_shapes	
:�*
dtype0�
6time_distributed_1/batch_normalization_2/batchnorm/mulMul<time_distributed_1/batch_normalization_2/batchnorm/Rsqrt:y:0Mtime_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
8time_distributed_1/batch_normalization_2/batchnorm/mul_1Mul-time_distributed_1/dense_2/Relu:activations:0:time_distributed_1/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Ctime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOptime_distributed_1_84912*
_output_shapes	
:�*
dtype0�
8time_distributed_1/batch_normalization_2/batchnorm/mul_2MulKtime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0:time_distributed_1/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
Ctime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOptime_distributed_1_84914*
_output_shapes	
:�*
dtype0�
6time_distributed_1/batch_normalization_2/batchnorm/subSubKtime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0<time_distributed_1/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
8time_distributed_1/batch_normalization_2/batchnorm/add_1AddV2<time_distributed_1/batch_normalization_2/batchnorm/mul_1:z:0:time_distributed_1/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
%time_distributed_1/dropout_2/IdentityIdentity<time_distributed_1/batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
0time_distributed_1/dense_3/MatMul/ReadVariableOpReadVariableOptime_distributed_1_84916* 
_output_shapes
:
��*
dtype0�
!time_distributed_1/dense_3/MatMulMatMul.time_distributed_1/dropout_2/Identity:output:08time_distributed_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
1time_distributed_1/dense_3/BiasAdd/ReadVariableOpReadVariableOptime_distributed_1_84918*
_output_shapes	
:�*
dtype0�
"time_distributed_1/dense_3/BiasAddBiasAdd+time_distributed_1/dense_3/MatMul:product:09time_distributed_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
time_distributed_1/dense_3/ReluRelu+time_distributed_1/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
$user_encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1time_distributed_1_84880time_distributed_1_84882time_distributed_1_84884time_distributed_1_84886time_distributed_1_84888time_distributed_1_84890time_distributed_1_84892time_distributed_1_84894time_distributed_1_84896time_distributed_1_84898time_distributed_1_84900time_distributed_1_84902time_distributed_1_84904time_distributed_1_84906time_distributed_1_84908time_distributed_1_84910time_distributed_1_84912time_distributed_1_84914time_distributed_1_84916time_distributed_1_84918user_encoder_85002user_encoder_85004user_encoder_85006user_encoder_85008user_encoder_85010user_encoder_85012*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*<
_read_only_resource_inputs
	
*<
config_proto,*

CPU

GPU 2J 8R(���������� *P
fKRI
G__inference_user_encoder_layer_call_and_return_conditional_losses_84507�
dot/PartitionedCallPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0-user_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *G
fBR@
>__inference_dot_layer_call_and_return_conditional_losses_84867�
activation/PartitionedCallPartitionedCalldot/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_84873{
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:�������������������

NoOpNoOp+^time_distributed_1/StatefulPartitionedCall@^time_distributed_1/batch_normalization/batchnorm/ReadVariableOpB^time_distributed_1/batch_normalization/batchnorm/ReadVariableOp_1B^time_distributed_1/batch_normalization/batchnorm/ReadVariableOp_2D^time_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOpB^time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOpD^time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_1D^time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_2F^time_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOpB^time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOpD^time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_1D^time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_2F^time_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOp0^time_distributed_1/dense/BiasAdd/ReadVariableOp/^time_distributed_1/dense/MatMul/ReadVariableOp2^time_distributed_1/dense_1/BiasAdd/ReadVariableOp1^time_distributed_1/dense_1/MatMul/ReadVariableOp2^time_distributed_1/dense_2/BiasAdd/ReadVariableOp1^time_distributed_1/dense_2/MatMul/ReadVariableOp2^time_distributed_1/dense_3/BiasAdd/ReadVariableOp1^time_distributed_1/dense_3/MatMul/ReadVariableOp%^user_encoder/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:���������#�:�������������������: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall2�
Atime_distributed_1/batch_normalization/batchnorm/ReadVariableOp_1Atime_distributed_1/batch_normalization/batchnorm/ReadVariableOp_12�
Atime_distributed_1/batch_normalization/batchnorm/ReadVariableOp_2Atime_distributed_1/batch_normalization/batchnorm/ReadVariableOp_22�
?time_distributed_1/batch_normalization/batchnorm/ReadVariableOp?time_distributed_1/batch_normalization/batchnorm/ReadVariableOp2�
Ctime_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOpCtime_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOp2�
Ctime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_1Ctime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_12�
Ctime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_2Ctime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp_22�
Atime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOpAtime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp2�
Etime_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOpEtime_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOp2�
Ctime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_1Ctime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_12�
Ctime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_2Ctime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp_22�
Atime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOpAtime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp2�
Etime_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOpEtime_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOp2b
/time_distributed_1/dense/BiasAdd/ReadVariableOp/time_distributed_1/dense/BiasAdd/ReadVariableOp2`
.time_distributed_1/dense/MatMul/ReadVariableOp.time_distributed_1/dense/MatMul/ReadVariableOp2f
1time_distributed_1/dense_1/BiasAdd/ReadVariableOp1time_distributed_1/dense_1/BiasAdd/ReadVariableOp2d
0time_distributed_1/dense_1/MatMul/ReadVariableOp0time_distributed_1/dense_1/MatMul/ReadVariableOp2f
1time_distributed_1/dense_2/BiasAdd/ReadVariableOp1time_distributed_1/dense_2/BiasAdd/ReadVariableOp2d
0time_distributed_1/dense_2/MatMul/ReadVariableOp0time_distributed_1/dense_2/MatMul/ReadVariableOp2f
1time_distributed_1/dense_3/BiasAdd/ReadVariableOp1time_distributed_1/dense_3/BiasAdd/ReadVariableOp2d
0time_distributed_1/dense_3/MatMul/ReadVariableOp0time_distributed_1/dense_3/MatMul/ReadVariableOp2L
$user_encoder/StatefulPartitionedCall$user_encoder/StatefulPartitionedCall:%!

_user_specified_name85012:%!

_user_specified_name85010:%!

_user_specified_name85008:%!

_user_specified_name85006:%!

_user_specified_name85004:%!

_user_specified_name85002:%!

_user_specified_name84918:%!

_user_specified_name84916:%!

_user_specified_name84914:%!

_user_specified_name84912:%!

_user_specified_name84910:%!

_user_specified_name84908:%!

_user_specified_name84906:%!

_user_specified_name84904:%!

_user_specified_name84902:%!

_user_specified_name84900:%!

_user_specified_name84898:%
!

_user_specified_name84896:%	!

_user_specified_name84894:%!

_user_specified_name84892:%!

_user_specified_name84890:%!

_user_specified_name84888:%!

_user_specified_name84886:%!

_user_specified_name84884:%!

_user_specified_name84882:%!

_user_specified_name84880:^Z
5
_output_shapes#
!:�������������������
!
_user_specified_name	input_2:U Q
,
_output_shapes
:���������#�
!
_user_specified_name	input_1
�
a
E__inference_activation_layer_call_and_return_conditional_losses_85573

inputs
identityU
SoftmaxSoftmaxinputs*
T0*0
_output_shapes
:������������������b
IdentityIdentitySoftmax:softmax:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:������������������:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�i
�
I__inference_self_attention_layer_call_and_return_conditional_losses_84247
qkvs

qkvs_1

qkvs_23
shape_1_readvariableop_resource:
��3
shape_4_readvariableop_resource:
��3
shape_7_readvariableop_resource:
��
identity��transpose/ReadVariableOp�transpose_2/ReadVariableOp�transpose_4/ReadVariableOpG
ShapeShapeqkvs*
T0*
_output_shapes
::��Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"�  �  S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  c
ReshapeReshapeqkvsReshape/shape:output:0*
T0*(
_output_shapes
:����������z
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       |
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0* 
_output_shapes
:
��`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"�  ����h
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
��i
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*(
_output_shapes
:����������S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#T
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:w
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*,
_output_shapes
:���������#�W
Shape_2ShapeReshape_2:output:0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape_2:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_3/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_3/shapePackReshape_3/shape/0:output:0strided_slice:output:0Reshape_3/shape/2:output:0Reshape_3/shape/3:output:0*
N*
T0*
_output_shapes
:|
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*/
_output_shapes
:���������#i
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
transpose_1	TransposeReshape_3:output:0transpose_1/perm:output:0*
T0*/
_output_shapes
:���������#K
Shape_3Shapeqkvs_1*
T0*
_output_shapes
::��U
	unstack_2UnpackShape_3:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_4/ReadVariableOpReadVariableOpshape_4_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Shape_4Const*
_output_shapes
:*
dtype0*
valueB"�  �  S
	unstack_3UnpackShape_4:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  i
	Reshape_4Reshapeqkvs_1Reshape_4/shape:output:0*
T0*(
_output_shapes
:����������|
transpose_2/ReadVariableOpReadVariableOpshape_4_readvariableop_resource* 
_output_shapes
:
��*
dtype0a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       �
transpose_2	Transpose"transpose_2/ReadVariableOp:value:0transpose_2/perm:output:0*
T0* 
_output_shapes
:
��`
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB"�  ����j
	Reshape_5Reshapetranspose_2:y:0Reshape_5/shape:output:0*
T0* 
_output_shapes
:
��m
MatMul_1MatMulReshape_4:output:0Reshape_5:output:0*
T0*(
_output_shapes
:����������S
Reshape_6/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#T
Reshape_6/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_6/shapePackunstack_2:output:0Reshape_6/shape/1:output:0Reshape_6/shape/2:output:0*
N*
T0*
_output_shapes
:y
	Reshape_6ReshapeMatMul_1:product:0Reshape_6/shape:output:0*
T0*,
_output_shapes
:���������#�W
Shape_5ShapeReshape_6:output:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_5:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
Reshape_7/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_7/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_7/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_7/shapePackReshape_7/shape/0:output:0strided_slice_1:output:0Reshape_7/shape/2:output:0Reshape_7/shape/3:output:0*
N*
T0*
_output_shapes
:|
	Reshape_7ReshapeReshape_6:output:0Reshape_7/shape:output:0*
T0*/
_output_shapes
:���������#i
transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
transpose_3	TransposeReshape_7:output:0transpose_3/perm:output:0*
T0*/
_output_shapes
:���������#K
Shape_6Shapeqkvs_2*
T0*
_output_shapes
::��U
	unstack_4UnpackShape_6:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_7/ReadVariableOpReadVariableOpshape_7_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Shape_7Const*
_output_shapes
:*
dtype0*
valueB"�  �  S
	unstack_5UnpackShape_7:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  i
	Reshape_8Reshapeqkvs_2Reshape_8/shape:output:0*
T0*(
_output_shapes
:����������|
transpose_4/ReadVariableOpReadVariableOpshape_7_readvariableop_resource* 
_output_shapes
:
��*
dtype0a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       �
transpose_4	Transpose"transpose_4/ReadVariableOp:value:0transpose_4/perm:output:0*
T0* 
_output_shapes
:
��`
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"�  ����j
	Reshape_9Reshapetranspose_4:y:0Reshape_9/shape:output:0*
T0* 
_output_shapes
:
��m
MatMul_2MatMulReshape_8:output:0Reshape_9:output:0*
T0*(
_output_shapes
:����������T
Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
value	B :#U
Reshape_10/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_10/shapePackunstack_4:output:0Reshape_10/shape/1:output:0Reshape_10/shape/2:output:0*
N*
T0*
_output_shapes
:{

Reshape_10ReshapeMatMul_2:product:0Reshape_10/shape:output:0*
T0*,
_output_shapes
:���������#�X
Shape_8ShapeReshape_10:output:0*
T0*
_output_shapes
::��_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_8:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
Reshape_11/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_11/shape/2Const*
_output_shapes
: *
dtype0*
value	B :T
Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_11/shapePackReshape_11/shape/0:output:0strided_slice_2:output:0Reshape_11/shape/2:output:0Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_11ReshapeReshape_10:output:0Reshape_11/shape:output:0*
T0*/
_output_shapes
:���������#i
transpose_5/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
transpose_5	TransposeReshape_11:output:0transpose_5/perm:output:0*
T0*/
_output_shapes
:���������#�
MatMul_3BatchMatMulV2transpose_1:y:0transpose_3:y:0*
T0*/
_output_shapes
:���������##*
adj_y(H
Cast/xConst*
_output_shapes
: *
dtype0*
value	B :M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    M
MaximumMaximumCast:y:0Const:output:0*
T0*
_output_shapes
: :
SqrtSqrtMaximum:z:0*
T0*
_output_shapes
: i
truedivRealDivMatMul_3:output:0Sqrt:y:0*
T0*/
_output_shapes
:���������##i
transpose_6/permConst*
_output_shapes
:*
dtype0*%
valueB"             z
transpose_6	Transposetruediv:z:0transpose_6/perm:output:0*
T0*/
_output_shapes
:���������##i
transpose_7/permConst*
_output_shapes
:*
dtype0*%
valueB"             ~
transpose_7	Transposetranspose_6:y:0transpose_7/perm:output:0*
T0*/
_output_shapes
:���������##]
SoftmaxSoftmaxtranspose_7:y:0*
T0*/
_output_shapes
:���������##�
MatMul_4BatchMatMulV2Softmax:softmax:0transpose_5:y:0*
T0*/
_output_shapes
:���������#*
adj_x(i
transpose_8/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
transpose_8	TransposeMatMul_4:output:0transpose_8/perm:output:0*
T0*/
_output_shapes
:���������#T
Shape_9Shapetranspose_8:y:0*
T0*
_output_shapes
::��_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_9:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
Reshape_12/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������U
Reshape_12/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_12/shapePackReshape_12/shape/0:output:0strided_slice_3:output:0Reshape_12/shape/2:output:0*
N*
T0*
_output_shapes
:x

Reshape_12Reshapetranspose_8:y:0Reshape_12/shape:output:0*
T0*,
_output_shapes
:���������#�g
IdentityIdentityReshape_12:output:0^NoOp*
T0*,
_output_shapes
:���������#�w
NoOpNoOp^transpose/ReadVariableOp^transpose_2/ReadVariableOp^transpose_4/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:���������#�:���������#�:���������#�: : : 24
transpose/ReadVariableOptranspose/ReadVariableOp28
transpose_2/ReadVariableOptranspose_2/ReadVariableOp28
transpose_4/ReadVariableOptranspose_4/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
,
_output_shapes
:���������#�

_user_specified_nameQKVs:RN
,
_output_shapes
:���������#�

_user_specified_nameQKVs:R N
,
_output_shapes
:���������#�

_user_specified_nameQKVs
�!
�
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_83603

inputs&
news_encoder_83557:
��!
news_encoder_83559:	�!
news_encoder_83561:	�!
news_encoder_83563:	�!
news_encoder_83565:	�!
news_encoder_83567:	�&
news_encoder_83569:
��!
news_encoder_83571:	�!
news_encoder_83573:	�!
news_encoder_83575:	�!
news_encoder_83577:	�!
news_encoder_83579:	�&
news_encoder_83581:
��!
news_encoder_83583:	�!
news_encoder_83585:	�!
news_encoder_83587:	�!
news_encoder_83589:	�!
news_encoder_83591:	�&
news_encoder_83593:
��!
news_encoder_83595:	�
identity��$news_encoder/StatefulPartitionedCallI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
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
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:�����������
$news_encoder/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0news_encoder_83557news_encoder_83559news_encoder_83561news_encoder_83563news_encoder_83565news_encoder_83567news_encoder_83569news_encoder_83571news_encoder_83573news_encoder_83575news_encoder_83577news_encoder_83579news_encoder_83581news_encoder_83583news_encoder_83585news_encoder_83587news_encoder_83589news_encoder_83591news_encoder_83593news_encoder_83595* 
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*0
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *P
fKRI
G__inference_news_encoder_layer_call_and_return_conditional_losses_83322\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape-news_encoder/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������I
NoOpNoOp%^news_encoder/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�������������������: : : : : : : : : : : : : : : : : : : : 2L
$news_encoder/StatefulPartitionedCall$news_encoder/StatefulPartitionedCall:%!

_user_specified_name83595:%!

_user_specified_name83593:%!

_user_specified_name83591:%!

_user_specified_name83589:%!

_user_specified_name83587:%!

_user_specified_name83585:%!

_user_specified_name83583:%!

_user_specified_name83581:%!

_user_specified_name83579:%!

_user_specified_name83577:%
!

_user_specified_name83575:%	!

_user_specified_name83573:%!

_user_specified_name83571:%!

_user_specified_name83569:%!

_user_specified_name83567:%!

_user_specified_name83565:%!

_user_specified_name83563:%!

_user_specified_name83561:%!

_user_specified_name83559:%!

_user_specified_name83557:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
*__inference_att_layer2_layer_call_fn_86067

inputs
unknown:
��
	unknown_0:	�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *N
fIRG
E__inference_att_layer2_layer_call_and_return_conditional_losses_84316p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:���������#�: : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name86063:%!

_user_specified_name86061:%!

_user_specified_name86059:T P
,
_output_shapes
:���������#�
 
_user_specified_nameinputs
�&
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_86409

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�:
�	
G__inference_news_encoder_layer_call_and_return_conditional_losses_83391
input_4
dense_83325:
��
dense_83327:	�(
batch_normalization_83330:	�(
batch_normalization_83332:	�(
batch_normalization_83334:	�(
batch_normalization_83336:	�!
dense_1_83345:
��
dense_1_83347:	�*
batch_normalization_1_83350:	�*
batch_normalization_1_83352:	�*
batch_normalization_1_83354:	�*
batch_normalization_1_83356:	�!
dense_2_83365:
��
dense_2_83367:	�*
batch_normalization_2_83370:	�*
batch_normalization_2_83372:	�*
batch_normalization_2_83374:	�*
batch_normalization_2_83376:	�!
dense_3_83385:
��
dense_3_83387:	�
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_83325dense_83327*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_83201�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_83330batch_normalization_83332batch_normalization_83334batch_normalization_83336*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_83002�
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_83343�
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_83345dense_1_83347*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_83239�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_83350batch_normalization_1_83352batch_normalization_1_83354batch_normalization_1_83356*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_83082�
dropout_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_83363�
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_83365dense_2_83367*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_83277�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_83370batch_normalization_2_83372batch_normalization_2_83374batch_normalization_2_83376*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_83162�
dropout_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_83383�
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_3_83385dense_3_83387*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_83315x
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:%!

_user_specified_name83387:%!

_user_specified_name83385:%!

_user_specified_name83376:%!

_user_specified_name83374:%!

_user_specified_name83372:%!

_user_specified_name83370:%!

_user_specified_name83367:%!

_user_specified_name83365:%!

_user_specified_name83356:%!

_user_specified_name83354:%
!

_user_specified_name83352:%	!

_user_specified_name83350:%!

_user_specified_name83347:%!

_user_specified_name83345:%!

_user_specified_name83336:%!

_user_specified_name83334:%!

_user_specified_name83332:%!

_user_specified_name83330:%!

_user_specified_name83327:%!

_user_specified_name83325:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_4
�
`
'__inference_dropout_layer_call_fn_86307

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_83227p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_85076
input_1
input_2
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:
��

unknown_20:
��

unknown_21:
��

unknown_22:
��

unknown_23:	�

unknown_24:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������*6
_read_only_resource_inputs
	*<
config_proto,*

CPU

GPU 2J 8R(���������� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_84876x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:���������#�:�������������������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name85072:%!

_user_specified_name85070:%!

_user_specified_name85068:%!

_user_specified_name85066:%!

_user_specified_name85064:%!

_user_specified_name85062:%!

_user_specified_name85060:%!

_user_specified_name85058:%!

_user_specified_name85056:%!

_user_specified_name85054:%!

_user_specified_name85052:%!

_user_specified_name85050:%!

_user_specified_name85048:%!

_user_specified_name85046:%!

_user_specified_name85044:%!

_user_specified_name85042:%!

_user_specified_name85040:%
!

_user_specified_name85038:%	!

_user_specified_name85036:%!

_user_specified_name85034:%!

_user_specified_name85032:%!

_user_specified_name85030:%!

_user_specified_name85028:%!

_user_specified_name85026:%!

_user_specified_name85024:%!

_user_specified_name85022:^Z
5
_output_shapes#
!:�������������������
!
_user_specified_name	input_2:U Q
,
_output_shapes
:���������#�
!
_user_specified_name	input_1
�

�
B__inference_dense_3_layer_call_and_return_conditional_losses_83315

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
я
�
K__inference_time_distributed_layer_call_and_return_conditional_losses_85820

inputsE
1news_encoder_dense_matmul_readvariableop_resource:
��A
2news_encoder_dense_biasadd_readvariableop_resource:	�W
Hnews_encoder_batch_normalization_assignmovingavg_readvariableop_resource:	�Y
Jnews_encoder_batch_normalization_assignmovingavg_1_readvariableop_resource:	�U
Fnews_encoder_batch_normalization_batchnorm_mul_readvariableop_resource:	�Q
Bnews_encoder_batch_normalization_batchnorm_readvariableop_resource:	�G
3news_encoder_dense_1_matmul_readvariableop_resource:
��C
4news_encoder_dense_1_biasadd_readvariableop_resource:	�Y
Jnews_encoder_batch_normalization_1_assignmovingavg_readvariableop_resource:	�[
Lnews_encoder_batch_normalization_1_assignmovingavg_1_readvariableop_resource:	�W
Hnews_encoder_batch_normalization_1_batchnorm_mul_readvariableop_resource:	�S
Dnews_encoder_batch_normalization_1_batchnorm_readvariableop_resource:	�G
3news_encoder_dense_2_matmul_readvariableop_resource:
��C
4news_encoder_dense_2_biasadd_readvariableop_resource:	�Y
Jnews_encoder_batch_normalization_2_assignmovingavg_readvariableop_resource:	�[
Lnews_encoder_batch_normalization_2_assignmovingavg_1_readvariableop_resource:	�W
Hnews_encoder_batch_normalization_2_batchnorm_mul_readvariableop_resource:	�S
Dnews_encoder_batch_normalization_2_batchnorm_readvariableop_resource:	�G
3news_encoder_dense_3_matmul_readvariableop_resource:
��C
4news_encoder_dense_3_biasadd_readvariableop_resource:	�
identity��0news_encoder/batch_normalization/AssignMovingAvg�?news_encoder/batch_normalization/AssignMovingAvg/ReadVariableOp�2news_encoder/batch_normalization/AssignMovingAvg_1�Anews_encoder/batch_normalization/AssignMovingAvg_1/ReadVariableOp�9news_encoder/batch_normalization/batchnorm/ReadVariableOp�=news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp�2news_encoder/batch_normalization_1/AssignMovingAvg�Anews_encoder/batch_normalization_1/AssignMovingAvg/ReadVariableOp�4news_encoder/batch_normalization_1/AssignMovingAvg_1�Cnews_encoder/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�;news_encoder/batch_normalization_1/batchnorm/ReadVariableOp�?news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp�2news_encoder/batch_normalization_2/AssignMovingAvg�Anews_encoder/batch_normalization_2/AssignMovingAvg/ReadVariableOp�4news_encoder/batch_normalization_2/AssignMovingAvg_1�Cnews_encoder/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�;news_encoder/batch_normalization_2/batchnorm/ReadVariableOp�?news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp�)news_encoder/dense/BiasAdd/ReadVariableOp�(news_encoder/dense/MatMul/ReadVariableOp�+news_encoder/dense_1/BiasAdd/ReadVariableOp�*news_encoder/dense_1/MatMul/ReadVariableOp�+news_encoder/dense_2/BiasAdd/ReadVariableOp�*news_encoder/dense_2/MatMul/ReadVariableOp�+news_encoder/dense_3/BiasAdd/ReadVariableOp�*news_encoder/dense_3/MatMul/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
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
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:�����������
(news_encoder/dense/MatMul/ReadVariableOpReadVariableOp1news_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
news_encoder/dense/MatMulMatMulReshape:output:00news_encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
)news_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp2news_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
news_encoder/dense/BiasAddBiasAdd#news_encoder/dense/MatMul:product:01news_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������w
news_encoder/dense/ReluRelu#news_encoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
?news_encoder/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
-news_encoder/batch_normalization/moments/meanMean%news_encoder/dense/Relu:activations:0Hnews_encoder/batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
5news_encoder/batch_normalization/moments/StopGradientStopGradient6news_encoder/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	��
:news_encoder/batch_normalization/moments/SquaredDifferenceSquaredDifference%news_encoder/dense/Relu:activations:0>news_encoder/batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
Cnews_encoder/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
1news_encoder/batch_normalization/moments/varianceMean>news_encoder/batch_normalization/moments/SquaredDifference:z:0Lnews_encoder/batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
0news_encoder/batch_normalization/moments/SqueezeSqueeze6news_encoder/batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
2news_encoder/batch_normalization/moments/Squeeze_1Squeeze:news_encoder/batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 {
6news_encoder/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
?news_encoder/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpHnews_encoder_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4news_encoder/batch_normalization/AssignMovingAvg/subSubGnews_encoder/batch_normalization/AssignMovingAvg/ReadVariableOp:value:09news_encoder/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
4news_encoder/batch_normalization/AssignMovingAvg/mulMul8news_encoder/batch_normalization/AssignMovingAvg/sub:z:0?news_encoder/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
0news_encoder/batch_normalization/AssignMovingAvgAssignSubVariableOpHnews_encoder_batch_normalization_assignmovingavg_readvariableop_resource8news_encoder/batch_normalization/AssignMovingAvg/mul:z:0@^news_encoder/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0}
8news_encoder/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Anews_encoder/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpJnews_encoder_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6news_encoder/batch_normalization/AssignMovingAvg_1/subSubInews_encoder/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0;news_encoder/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
6news_encoder/batch_normalization/AssignMovingAvg_1/mulMul:news_encoder/batch_normalization/AssignMovingAvg_1/sub:z:0Anews_encoder/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization/AssignMovingAvg_1AssignSubVariableOpJnews_encoder_batch_normalization_assignmovingavg_1_readvariableop_resource:news_encoder/batch_normalization/AssignMovingAvg_1/mul:z:0B^news_encoder/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0u
0news_encoder/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
.news_encoder/batch_normalization/batchnorm/addAddV2;news_encoder/batch_normalization/moments/Squeeze_1:output:09news_encoder/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
0news_encoder/batch_normalization/batchnorm/RsqrtRsqrt2news_encoder/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
=news_encoder/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpFnews_encoder_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.news_encoder/batch_normalization/batchnorm/mulMul4news_encoder/batch_normalization/batchnorm/Rsqrt:y:0Enews_encoder/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
0news_encoder/batch_normalization/batchnorm/mul_1Mul%news_encoder/dense/Relu:activations:02news_encoder/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
0news_encoder/batch_normalization/batchnorm/mul_2Mul9news_encoder/batch_normalization/moments/Squeeze:output:02news_encoder/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
9news_encoder/batch_normalization/batchnorm/ReadVariableOpReadVariableOpBnews_encoder_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.news_encoder/batch_normalization/batchnorm/subSubAnews_encoder/batch_normalization/batchnorm/ReadVariableOp:value:04news_encoder/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
0news_encoder/batch_normalization/batchnorm/add_1AddV24news_encoder/batch_normalization/batchnorm/mul_1:z:02news_encoder/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������g
"news_encoder/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
 news_encoder/dropout/dropout/MulMul4news_encoder/batch_normalization/batchnorm/add_1:z:0+news_encoder/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
"news_encoder/dropout/dropout/ShapeShape4news_encoder/batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
9news_encoder/dropout/dropout/random_uniform/RandomUniformRandomUniform+news_encoder/dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*p
+news_encoder/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
)news_encoder/dropout/dropout/GreaterEqualGreaterEqualBnews_encoder/dropout/dropout/random_uniform/RandomUniform:output:04news_encoder/dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������i
$news_encoder/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
%news_encoder/dropout/dropout/SelectV2SelectV2-news_encoder/dropout/dropout/GreaterEqual:z:0$news_encoder/dropout/dropout/Mul:z:0-news_encoder/dropout/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
*news_encoder/dense_1/MatMul/ReadVariableOpReadVariableOp3news_encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
news_encoder/dense_1/MatMulMatMul.news_encoder/dropout/dropout/SelectV2:output:02news_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+news_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp4news_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
news_encoder/dense_1/BiasAddBiasAdd%news_encoder/dense_1/MatMul:product:03news_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
news_encoder/dense_1/ReluRelu%news_encoder/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Anews_encoder/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
/news_encoder/batch_normalization_1/moments/meanMean'news_encoder/dense_1/Relu:activations:0Jnews_encoder/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
7news_encoder/batch_normalization_1/moments/StopGradientStopGradient8news_encoder/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	��
<news_encoder/batch_normalization_1/moments/SquaredDifferenceSquaredDifference'news_encoder/dense_1/Relu:activations:0@news_encoder/batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
Enews_encoder/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
3news_encoder/batch_normalization_1/moments/varianceMean@news_encoder/batch_normalization_1/moments/SquaredDifference:z:0Nnews_encoder/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
2news_encoder/batch_normalization_1/moments/SqueezeSqueeze8news_encoder/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
4news_encoder/batch_normalization_1/moments/Squeeze_1Squeeze<news_encoder/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 }
8news_encoder/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Anews_encoder/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpJnews_encoder_batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6news_encoder/batch_normalization_1/AssignMovingAvg/subSubInews_encoder/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0;news_encoder/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
6news_encoder/batch_normalization_1/AssignMovingAvg/mulMul:news_encoder/batch_normalization_1/AssignMovingAvg/sub:z:0Anews_encoder/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_1/AssignMovingAvgAssignSubVariableOpJnews_encoder_batch_normalization_1_assignmovingavg_readvariableop_resource:news_encoder/batch_normalization_1/AssignMovingAvg/mul:z:0B^news_encoder/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0
:news_encoder/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Cnews_encoder/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpLnews_encoder_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8news_encoder/batch_normalization_1/AssignMovingAvg_1/subSubKnews_encoder/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0=news_encoder/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
8news_encoder/batch_normalization_1/AssignMovingAvg_1/mulMul<news_encoder/batch_normalization_1/AssignMovingAvg_1/sub:z:0Cnews_encoder/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
4news_encoder/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOpLnews_encoder_batch_normalization_1_assignmovingavg_1_readvariableop_resource<news_encoder/batch_normalization_1/AssignMovingAvg_1/mul:z:0D^news_encoder/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0w
2news_encoder/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
0news_encoder/batch_normalization_1/batchnorm/addAddV2=news_encoder/batch_normalization_1/moments/Squeeze_1:output:0;news_encoder/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_1/batchnorm/RsqrtRsqrt4news_encoder/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
?news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpHnews_encoder_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization_1/batchnorm/mulMul6news_encoder/batch_normalization_1/batchnorm/Rsqrt:y:0Gnews_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_1/batchnorm/mul_1Mul'news_encoder/dense_1/Relu:activations:04news_encoder/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
2news_encoder/batch_normalization_1/batchnorm/mul_2Mul;news_encoder/batch_normalization_1/moments/Squeeze:output:04news_encoder/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
;news_encoder/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpDnews_encoder_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization_1/batchnorm/subSubCnews_encoder/batch_normalization_1/batchnorm/ReadVariableOp:value:06news_encoder/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_1/batchnorm/add_1AddV26news_encoder/batch_normalization_1/batchnorm/mul_1:z:04news_encoder/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������i
$news_encoder/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"news_encoder/dropout_1/dropout/MulMul6news_encoder/batch_normalization_1/batchnorm/add_1:z:0-news_encoder/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
$news_encoder/dropout_1/dropout/ShapeShape6news_encoder/batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
;news_encoder/dropout_1/dropout/random_uniform/RandomUniformRandomUniform-news_encoder/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed2*

seed*r
-news_encoder/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
+news_encoder/dropout_1/dropout/GreaterEqualGreaterEqualDnews_encoder/dropout_1/dropout/random_uniform/RandomUniform:output:06news_encoder/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������k
&news_encoder/dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
'news_encoder/dropout_1/dropout/SelectV2SelectV2/news_encoder/dropout_1/dropout/GreaterEqual:z:0&news_encoder/dropout_1/dropout/Mul:z:0/news_encoder/dropout_1/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
*news_encoder/dense_2/MatMul/ReadVariableOpReadVariableOp3news_encoder_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
news_encoder/dense_2/MatMulMatMul0news_encoder/dropout_1/dropout/SelectV2:output:02news_encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+news_encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp4news_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
news_encoder/dense_2/BiasAddBiasAdd%news_encoder/dense_2/MatMul:product:03news_encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
news_encoder/dense_2/ReluRelu%news_encoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Anews_encoder/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
/news_encoder/batch_normalization_2/moments/meanMean'news_encoder/dense_2/Relu:activations:0Jnews_encoder/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
7news_encoder/batch_normalization_2/moments/StopGradientStopGradient8news_encoder/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	��
<news_encoder/batch_normalization_2/moments/SquaredDifferenceSquaredDifference'news_encoder/dense_2/Relu:activations:0@news_encoder/batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
Enews_encoder/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
3news_encoder/batch_normalization_2/moments/varianceMean@news_encoder/batch_normalization_2/moments/SquaredDifference:z:0Nnews_encoder/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
2news_encoder/batch_normalization_2/moments/SqueezeSqueeze8news_encoder/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
4news_encoder/batch_normalization_2/moments/Squeeze_1Squeeze<news_encoder/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 }
8news_encoder/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Anews_encoder/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpJnews_encoder_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6news_encoder/batch_normalization_2/AssignMovingAvg/subSubInews_encoder/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0;news_encoder/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
6news_encoder/batch_normalization_2/AssignMovingAvg/mulMul:news_encoder/batch_normalization_2/AssignMovingAvg/sub:z:0Anews_encoder/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_2/AssignMovingAvgAssignSubVariableOpJnews_encoder_batch_normalization_2_assignmovingavg_readvariableop_resource:news_encoder/batch_normalization_2/AssignMovingAvg/mul:z:0B^news_encoder/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0
:news_encoder/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Cnews_encoder/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpLnews_encoder_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8news_encoder/batch_normalization_2/AssignMovingAvg_1/subSubKnews_encoder/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0=news_encoder/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
8news_encoder/batch_normalization_2/AssignMovingAvg_1/mulMul<news_encoder/batch_normalization_2/AssignMovingAvg_1/sub:z:0Cnews_encoder/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
4news_encoder/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpLnews_encoder_batch_normalization_2_assignmovingavg_1_readvariableop_resource<news_encoder/batch_normalization_2/AssignMovingAvg_1/mul:z:0D^news_encoder/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0w
2news_encoder/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
0news_encoder/batch_normalization_2/batchnorm/addAddV2=news_encoder/batch_normalization_2/moments/Squeeze_1:output:0;news_encoder/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_2/batchnorm/RsqrtRsqrt4news_encoder/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
?news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpHnews_encoder_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization_2/batchnorm/mulMul6news_encoder/batch_normalization_2/batchnorm/Rsqrt:y:0Gnews_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_2/batchnorm/mul_1Mul'news_encoder/dense_2/Relu:activations:04news_encoder/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
2news_encoder/batch_normalization_2/batchnorm/mul_2Mul;news_encoder/batch_normalization_2/moments/Squeeze:output:04news_encoder/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
;news_encoder/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpDnews_encoder_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization_2/batchnorm/subSubCnews_encoder/batch_normalization_2/batchnorm/ReadVariableOp:value:06news_encoder/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_2/batchnorm/add_1AddV26news_encoder/batch_normalization_2/batchnorm/mul_1:z:04news_encoder/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������i
$news_encoder/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"news_encoder/dropout_2/dropout/MulMul6news_encoder/batch_normalization_2/batchnorm/add_1:z:0-news_encoder/dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
$news_encoder/dropout_2/dropout/ShapeShape6news_encoder/batch_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
;news_encoder/dropout_2/dropout/random_uniform/RandomUniformRandomUniform-news_encoder/dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed2*

seed*r
-news_encoder/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
+news_encoder/dropout_2/dropout/GreaterEqualGreaterEqualDnews_encoder/dropout_2/dropout/random_uniform/RandomUniform:output:06news_encoder/dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������k
&news_encoder/dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
'news_encoder/dropout_2/dropout/SelectV2SelectV2/news_encoder/dropout_2/dropout/GreaterEqual:z:0&news_encoder/dropout_2/dropout/Mul:z:0/news_encoder/dropout_2/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
*news_encoder/dense_3/MatMul/ReadVariableOpReadVariableOp3news_encoder_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
news_encoder/dense_3/MatMulMatMul0news_encoder/dropout_2/dropout/SelectV2:output:02news_encoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+news_encoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp4news_encoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
news_encoder/dense_3/BiasAddBiasAdd%news_encoder/dense_3/MatMul:product:03news_encoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
news_encoder/dense_3/ReluRelu%news_encoder/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape'news_encoder/dense_3/Relu:activations:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:��������������������
NoOpNoOp1^news_encoder/batch_normalization/AssignMovingAvg@^news_encoder/batch_normalization/AssignMovingAvg/ReadVariableOp3^news_encoder/batch_normalization/AssignMovingAvg_1B^news_encoder/batch_normalization/AssignMovingAvg_1/ReadVariableOp:^news_encoder/batch_normalization/batchnorm/ReadVariableOp>^news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp3^news_encoder/batch_normalization_1/AssignMovingAvgB^news_encoder/batch_normalization_1/AssignMovingAvg/ReadVariableOp5^news_encoder/batch_normalization_1/AssignMovingAvg_1D^news_encoder/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp<^news_encoder/batch_normalization_1/batchnorm/ReadVariableOp@^news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp3^news_encoder/batch_normalization_2/AssignMovingAvgB^news_encoder/batch_normalization_2/AssignMovingAvg/ReadVariableOp5^news_encoder/batch_normalization_2/AssignMovingAvg_1D^news_encoder/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp<^news_encoder/batch_normalization_2/batchnorm/ReadVariableOp@^news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp*^news_encoder/dense/BiasAdd/ReadVariableOp)^news_encoder/dense/MatMul/ReadVariableOp,^news_encoder/dense_1/BiasAdd/ReadVariableOp+^news_encoder/dense_1/MatMul/ReadVariableOp,^news_encoder/dense_2/BiasAdd/ReadVariableOp+^news_encoder/dense_2/MatMul/ReadVariableOp,^news_encoder/dense_3/BiasAdd/ReadVariableOp+^news_encoder/dense_3/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�������������������: : : : : : : : : : : : : : : : : : : : 2�
?news_encoder/batch_normalization/AssignMovingAvg/ReadVariableOp?news_encoder/batch_normalization/AssignMovingAvg/ReadVariableOp2�
Anews_encoder/batch_normalization/AssignMovingAvg_1/ReadVariableOpAnews_encoder/batch_normalization/AssignMovingAvg_1/ReadVariableOp2h
2news_encoder/batch_normalization/AssignMovingAvg_12news_encoder/batch_normalization/AssignMovingAvg_12d
0news_encoder/batch_normalization/AssignMovingAvg0news_encoder/batch_normalization/AssignMovingAvg2v
9news_encoder/batch_normalization/batchnorm/ReadVariableOp9news_encoder/batch_normalization/batchnorm/ReadVariableOp2~
=news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp=news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp2�
Anews_encoder/batch_normalization_1/AssignMovingAvg/ReadVariableOpAnews_encoder/batch_normalization_1/AssignMovingAvg/ReadVariableOp2�
Cnews_encoder/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpCnews_encoder/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2l
4news_encoder/batch_normalization_1/AssignMovingAvg_14news_encoder/batch_normalization_1/AssignMovingAvg_12h
2news_encoder/batch_normalization_1/AssignMovingAvg2news_encoder/batch_normalization_1/AssignMovingAvg2z
;news_encoder/batch_normalization_1/batchnorm/ReadVariableOp;news_encoder/batch_normalization_1/batchnorm/ReadVariableOp2�
?news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp?news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp2�
Anews_encoder/batch_normalization_2/AssignMovingAvg/ReadVariableOpAnews_encoder/batch_normalization_2/AssignMovingAvg/ReadVariableOp2�
Cnews_encoder/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpCnews_encoder/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2l
4news_encoder/batch_normalization_2/AssignMovingAvg_14news_encoder/batch_normalization_2/AssignMovingAvg_12h
2news_encoder/batch_normalization_2/AssignMovingAvg2news_encoder/batch_normalization_2/AssignMovingAvg2z
;news_encoder/batch_normalization_2/batchnorm/ReadVariableOp;news_encoder/batch_normalization_2/batchnorm/ReadVariableOp2�
?news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp?news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp2V
)news_encoder/dense/BiasAdd/ReadVariableOp)news_encoder/dense/BiasAdd/ReadVariableOp2T
(news_encoder/dense/MatMul/ReadVariableOp(news_encoder/dense/MatMul/ReadVariableOp2Z
+news_encoder/dense_1/BiasAdd/ReadVariableOp+news_encoder/dense_1/BiasAdd/ReadVariableOp2X
*news_encoder/dense_1/MatMul/ReadVariableOp*news_encoder/dense_1/MatMul/ReadVariableOp2Z
+news_encoder/dense_2/BiasAdd/ReadVariableOp+news_encoder/dense_2/BiasAdd/ReadVariableOp2X
*news_encoder/dense_2/MatMul/ReadVariableOp*news_encoder/dense_2/MatMul/ReadVariableOp2Z
+news_encoder/dense_3/BiasAdd/ReadVariableOp+news_encoder/dense_3/BiasAdd/ReadVariableOp2X
*news_encoder/dense_3/MatMul/ReadVariableOp*news_encoder/dense_3/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
0__inference_time_distributed_layer_call_fn_85663

inputs
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
��

unknown_18:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*6
_read_only_resource_inputs
	
*<
config_proto,*

CPU

GPU 2J 8R(���������� *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_83861}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�������������������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name85659:%!

_user_specified_name85657:%!

_user_specified_name85655:%!

_user_specified_name85653:%!

_user_specified_name85651:%!

_user_specified_name85649:%!

_user_specified_name85647:%!

_user_specified_name85645:%!

_user_specified_name85643:%!

_user_specified_name85641:%
!

_user_specified_name85639:%	!

_user_specified_name85637:%!

_user_specified_name85635:%!

_user_specified_name85633:%!

_user_specified_name85631:%!

_user_specified_name85629:%!

_user_specified_name85627:%!

_user_specified_name85625:%!

_user_specified_name85623:%!

_user_specified_name85621:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
��
�/
!__inference__traced_restore_87301
file_prefix1
assignvariableop_dense_kernel:
��,
assignvariableop_1_dense_bias:	�;
,assignvariableop_2_batch_normalization_gamma:	�:
+assignvariableop_3_batch_normalization_beta:	�5
!assignvariableop_4_dense_1_kernel:
��.
assignvariableop_5_dense_1_bias:	�=
.assignvariableop_6_batch_normalization_1_gamma:	�<
-assignvariableop_7_batch_normalization_1_beta:	�5
!assignvariableop_8_dense_2_kernel:
��.
assignvariableop_9_dense_2_bias:	�>
/assignvariableop_10_batch_normalization_2_gamma:	�=
.assignvariableop_11_batch_normalization_2_beta:	�6
"assignvariableop_12_dense_3_kernel:
��/
 assignvariableop_13_dense_3_bias:	�B
3assignvariableop_14_batch_normalization_moving_mean:	�F
7assignvariableop_15_batch_normalization_moving_variance:	�D
5assignvariableop_16_batch_normalization_1_moving_mean:	�H
9assignvariableop_17_batch_normalization_1_moving_variance:	�D
5assignvariableop_18_batch_normalization_2_moving_mean:	�H
9assignvariableop_19_batch_normalization_2_moving_variance:	�9
%assignvariableop_20_self_attention_wq:
��9
%assignvariableop_21_self_attention_wk:
��9
%assignvariableop_22_self_attention_wv:
��4
 assignvariableop_23_att_layer2_w:
��/
 assignvariableop_24_att_layer2_b:	�3
 assignvariableop_25_att_layer2_q:	�'
assignvariableop_26_iteration:	 +
!assignvariableop_27_learning_rate: ;
'assignvariableop_28_adam_m_dense_kernel:
��;
'assignvariableop_29_adam_v_dense_kernel:
��4
%assignvariableop_30_adam_m_dense_bias:	�4
%assignvariableop_31_adam_v_dense_bias:	�C
4assignvariableop_32_adam_m_batch_normalization_gamma:	�C
4assignvariableop_33_adam_v_batch_normalization_gamma:	�B
3assignvariableop_34_adam_m_batch_normalization_beta:	�B
3assignvariableop_35_adam_v_batch_normalization_beta:	�=
)assignvariableop_36_adam_m_dense_1_kernel:
��=
)assignvariableop_37_adam_v_dense_1_kernel:
��6
'assignvariableop_38_adam_m_dense_1_bias:	�6
'assignvariableop_39_adam_v_dense_1_bias:	�E
6assignvariableop_40_adam_m_batch_normalization_1_gamma:	�E
6assignvariableop_41_adam_v_batch_normalization_1_gamma:	�D
5assignvariableop_42_adam_m_batch_normalization_1_beta:	�D
5assignvariableop_43_adam_v_batch_normalization_1_beta:	�=
)assignvariableop_44_adam_m_dense_2_kernel:
��=
)assignvariableop_45_adam_v_dense_2_kernel:
��6
'assignvariableop_46_adam_m_dense_2_bias:	�6
'assignvariableop_47_adam_v_dense_2_bias:	�E
6assignvariableop_48_adam_m_batch_normalization_2_gamma:	�E
6assignvariableop_49_adam_v_batch_normalization_2_gamma:	�D
5assignvariableop_50_adam_m_batch_normalization_2_beta:	�D
5assignvariableop_51_adam_v_batch_normalization_2_beta:	�=
)assignvariableop_52_adam_m_dense_3_kernel:
��=
)assignvariableop_53_adam_v_dense_3_kernel:
��6
'assignvariableop_54_adam_m_dense_3_bias:	�6
'assignvariableop_55_adam_v_dense_3_bias:	�@
,assignvariableop_56_adam_m_self_attention_wq:
��@
,assignvariableop_57_adam_v_self_attention_wq:
��@
,assignvariableop_58_adam_m_self_attention_wk:
��@
,assignvariableop_59_adam_v_self_attention_wk:
��@
,assignvariableop_60_adam_m_self_attention_wv:
��@
,assignvariableop_61_adam_v_self_attention_wv:
��;
'assignvariableop_62_adam_m_att_layer2_w:
��;
'assignvariableop_63_adam_v_att_layer2_w:
��6
'assignvariableop_64_adam_m_att_layer2_b:	�6
'assignvariableop_65_adam_v_att_layer2_b:	�:
'assignvariableop_66_adam_m_att_layer2_q:	�:
'assignvariableop_67_adam_v_att_layer2_q:	�#
assignvariableop_68_total: #
assignvariableop_69_count: 1
"assignvariableop_70_true_positives:	�1
"assignvariableop_71_true_negatives:	�2
#assignvariableop_72_false_positives:	�2
#assignvariableop_73_false_negatives:	�
identity_75��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*�
value�B�KB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*�
value�B�KB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Y
dtypesO
M2K	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_1_betaIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_2_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_2_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_2_gammaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_2_betaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_3_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_3_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp3assignvariableop_14_batch_normalization_moving_meanIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp7assignvariableop_15_batch_normalization_moving_varianceIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_1_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_1_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_2_moving_meanIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_2_moving_varianceIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp%assignvariableop_20_self_attention_wqIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp%assignvariableop_21_self_attention_wkIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp%assignvariableop_22_self_attention_wvIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp assignvariableop_23_att_layer2_wIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp assignvariableop_24_att_layer2_bIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp assignvariableop_25_att_layer2_qIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_iterationIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp!assignvariableop_27_learning_rateIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_m_dense_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_v_dense_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_m_dense_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp%assignvariableop_31_adam_v_dense_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_m_batch_normalization_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_v_batch_normalization_gammaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp3assignvariableop_34_adam_m_batch_normalization_betaIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp3assignvariableop_35_adam_v_batch_normalization_betaIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_m_dense_1_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_v_dense_1_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_m_dense_1_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_v_dense_1_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp6assignvariableop_40_adam_m_batch_normalization_1_gammaIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_v_batch_normalization_1_gammaIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_m_batch_normalization_1_betaIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp5assignvariableop_43_adam_v_batch_normalization_1_betaIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_m_dense_2_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_v_dense_2_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_m_dense_2_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp'assignvariableop_47_adam_v_dense_2_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp6assignvariableop_48_adam_m_batch_normalization_2_gammaIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_v_batch_normalization_2_gammaIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_m_batch_normalization_2_betaIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adam_v_batch_normalization_2_betaIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_m_dense_3_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_v_dense_3_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_m_dense_3_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp'assignvariableop_55_adam_v_dense_3_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp,assignvariableop_56_adam_m_self_attention_wqIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp,assignvariableop_57_adam_v_self_attention_wqIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp,assignvariableop_58_adam_m_self_attention_wkIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp,assignvariableop_59_adam_v_self_attention_wkIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp,assignvariableop_60_adam_m_self_attention_wvIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp,assignvariableop_61_adam_v_self_attention_wvIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp'assignvariableop_62_adam_m_att_layer2_wIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp'assignvariableop_63_adam_v_att_layer2_wIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp'assignvariableop_64_adam_m_att_layer2_bIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp'assignvariableop_65_adam_v_att_layer2_bIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp'assignvariableop_66_adam_m_att_layer2_qIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp'assignvariableop_67_adam_v_att_layer2_qIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOpassignvariableop_68_totalIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOpassignvariableop_69_countIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp"assignvariableop_70_true_positivesIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp"assignvariableop_71_true_negativesIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp#assignvariableop_72_false_positivesIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp#assignvariableop_73_false_negativesIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_74Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_75IdentityIdentity_74:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_75Identity_75:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:/J+
)
_user_specified_namefalse_negatives:/I+
)
_user_specified_namefalse_positives:.H*
(
_user_specified_nametrue_negatives:.G*
(
_user_specified_nametrue_positives:%F!

_user_specified_namecount:%E!

_user_specified_nametotal:3D/
-
_user_specified_nameAdam/v/att_layer2/q:3C/
-
_user_specified_nameAdam/m/att_layer2/q:3B/
-
_user_specified_nameAdam/v/att_layer2/b:3A/
-
_user_specified_nameAdam/m/att_layer2/b:3@/
-
_user_specified_nameAdam/v/att_layer2/W:3?/
-
_user_specified_nameAdam/m/att_layer2/W:8>4
2
_user_specified_nameAdam/v/self_attention/WV:8=4
2
_user_specified_nameAdam/m/self_attention/WV:8<4
2
_user_specified_nameAdam/v/self_attention/WK:8;4
2
_user_specified_nameAdam/m/self_attention/WK:8:4
2
_user_specified_nameAdam/v/self_attention/WQ:894
2
_user_specified_nameAdam/m/self_attention/WQ:38/
-
_user_specified_nameAdam/v/dense_3/bias:37/
-
_user_specified_nameAdam/m/dense_3/bias:561
/
_user_specified_nameAdam/v/dense_3/kernel:551
/
_user_specified_nameAdam/m/dense_3/kernel:A4=
;
_user_specified_name#!Adam/v/batch_normalization_2/beta:A3=
;
_user_specified_name#!Adam/m/batch_normalization_2/beta:B2>
<
_user_specified_name$"Adam/v/batch_normalization_2/gamma:B1>
<
_user_specified_name$"Adam/m/batch_normalization_2/gamma:30/
-
_user_specified_nameAdam/v/dense_2/bias:3//
-
_user_specified_nameAdam/m/dense_2/bias:5.1
/
_user_specified_nameAdam/v/dense_2/kernel:5-1
/
_user_specified_nameAdam/m/dense_2/kernel:A,=
;
_user_specified_name#!Adam/v/batch_normalization_1/beta:A+=
;
_user_specified_name#!Adam/m/batch_normalization_1/beta:B*>
<
_user_specified_name$"Adam/v/batch_normalization_1/gamma:B)>
<
_user_specified_name$"Adam/m/batch_normalization_1/gamma:3(/
-
_user_specified_nameAdam/v/dense_1/bias:3'/
-
_user_specified_nameAdam/m/dense_1/bias:5&1
/
_user_specified_nameAdam/v/dense_1/kernel:5%1
/
_user_specified_nameAdam/m/dense_1/kernel:?$;
9
_user_specified_name!Adam/v/batch_normalization/beta:?#;
9
_user_specified_name!Adam/m/batch_normalization/beta:@"<
:
_user_specified_name" Adam/v/batch_normalization/gamma:@!<
:
_user_specified_name" Adam/m/batch_normalization/gamma:1 -
+
_user_specified_nameAdam/v/dense/bias:1-
+
_user_specified_nameAdam/m/dense/bias:3/
-
_user_specified_nameAdam/v/dense/kernel:3/
-
_user_specified_nameAdam/m/dense/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_nameatt_layer2/q:,(
&
_user_specified_nameatt_layer2/b:,(
&
_user_specified_nameatt_layer2/W:1-
+
_user_specified_nameself_attention/WV:1-
+
_user_specified_nameself_attention/WK:1-
+
_user_specified_nameself_attention/WQ:EA
?
_user_specified_name'%batch_normalization_2/moving_variance:A=
;
_user_specified_name#!batch_normalization_2/moving_mean:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:A=
;
_user_specified_name#!batch_normalization_1/moving_mean:C?
=
_user_specified_name%#batch_normalization/moving_variance:?;
9
_user_specified_name!batch_normalization/moving_mean:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel::6
4
_user_specified_namebatch_normalization_2/beta:;7
5
_user_specified_namebatch_normalization_2/gamma:,
(
&
_user_specified_namedense_2/bias:.	*
(
_user_specified_namedense_2/kernel::6
4
_user_specified_namebatch_normalization_1/beta:;7
5
_user_specified_namebatch_normalization_1/gamma:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:84
2
_user_specified_namebatch_normalization/beta:95
3
_user_specified_namebatch_normalization/gamma:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�&
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_83142

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
)__inference_dropout_2_layer_call_fn_86561

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_83303p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
)__inference_dropout_1_layer_call_fn_86434

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_83265p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_att_layer2_layer_call_fn_86078

inputs
unknown:
��
	unknown_0:	�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *N
fIRG
E__inference_att_layer2_layer_call_and_return_conditional_losses_84498p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:���������#�: : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name86074:%!

_user_specified_name86072:%!

_user_specified_name86070:T P
,
_output_shapes
:���������#�
 
_user_specified_nameinputs
�	
�
5__inference_batch_normalization_2_layer_call_fn_86502

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_83162p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name86498:%!

_user_specified_name86496:%!

_user_specified_name86494:%!

_user_specified_name86492:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
@__inference_dense_layer_call_and_return_conditional_losses_86222

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�>
�	
G__inference_news_encoder_layer_call_and_return_conditional_losses_83322
input_4
dense_83202:
��
dense_83204:	�(
batch_normalization_83207:	�(
batch_normalization_83209:	�(
batch_normalization_83211:	�(
batch_normalization_83213:	�!
dense_1_83240:
��
dense_1_83242:	�*
batch_normalization_1_83245:	�*
batch_normalization_1_83247:	�*
batch_normalization_1_83249:	�*
batch_normalization_1_83251:	�!
dense_2_83278:
��
dense_2_83280:	�*
batch_normalization_2_83283:	�*
batch_normalization_2_83285:	�*
batch_normalization_2_83287:	�*
batch_normalization_2_83289:	�!
dense_3_83316:
��
dense_3_83318:	�
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_83202dense_83204*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_83201�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_83207batch_normalization_83209batch_normalization_83211batch_normalization_83213*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_82982�
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_83227�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_83240dense_1_83242*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_83239�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_83245batch_normalization_1_83247batch_normalization_1_83249batch_normalization_1_83251*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_83062�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_83265�
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_83278dense_2_83280*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_83277�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_83283batch_normalization_2_83285batch_normalization_2_83287batch_normalization_2_83289*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_83142�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_83303�
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_3_83316dense_3_83318*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_83315x
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:%!

_user_specified_name83318:%!

_user_specified_name83316:%!

_user_specified_name83289:%!

_user_specified_name83287:%!

_user_specified_name83285:%!

_user_specified_name83283:%!

_user_specified_name83280:%!

_user_specified_name83278:%!

_user_specified_name83251:%!

_user_specified_name83249:%
!

_user_specified_name83247:%	!

_user_specified_name83245:%!

_user_specified_name83242:%!

_user_specified_name83240:%!

_user_specified_name83213:%!

_user_specified_name83211:%!

_user_specified_name83209:%!

_user_specified_name83207:%!

_user_specified_name83204:%!

_user_specified_name83202:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_4
�	
�
5__inference_batch_normalization_1_layer_call_fn_86375

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_83082p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name86371:%!

_user_specified_name86369:%!

_user_specified_name86367:%!

_user_specified_name86365:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_86583

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_user_encoder_layer_call_fn_84621
input_5
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:
��

unknown_20:
��

unknown_21:
��

unknown_22:
��

unknown_23:	�

unknown_24:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*<
_read_only_resource_inputs
	
*<
config_proto,*

CPU

GPU 2J 8R(���������� *P
fKRI
G__inference_user_encoder_layer_call_and_return_conditional_losses_84507p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:���������#�: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name84617:%!

_user_specified_name84615:%!

_user_specified_name84613:%!

_user_specified_name84611:%!

_user_specified_name84609:%!

_user_specified_name84607:%!

_user_specified_name84605:%!

_user_specified_name84603:%!

_user_specified_name84601:%!

_user_specified_name84599:%!

_user_specified_name84597:%!

_user_specified_name84595:%!

_user_specified_name84593:%!

_user_specified_name84591:%!

_user_specified_name84589:%!

_user_specified_name84587:%
!

_user_specified_name84585:%	!

_user_specified_name84583:%!

_user_specified_name84581:%!

_user_specified_name84579:%!

_user_specified_name84577:%!

_user_specified_name84575:%!

_user_specified_name84573:%!

_user_specified_name84571:%!

_user_specified_name84569:%!

_user_specified_name84567:U Q
,
_output_shapes
:���������#�
!
_user_specified_name	input_5
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_86429

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_dense_3_layer_call_fn_86592

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_83315p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name86588:%!

_user_specified_name86586:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_83383

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_86556

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
5__inference_batch_normalization_2_layer_call_fn_86489

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_83142p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name86485:%!

_user_specified_name86483:%!

_user_specified_name86481:%!

_user_specified_name86479:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
ӏ
�
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_85453

inputsE
1news_encoder_dense_matmul_readvariableop_resource:
��A
2news_encoder_dense_biasadd_readvariableop_resource:	�W
Hnews_encoder_batch_normalization_assignmovingavg_readvariableop_resource:	�Y
Jnews_encoder_batch_normalization_assignmovingavg_1_readvariableop_resource:	�U
Fnews_encoder_batch_normalization_batchnorm_mul_readvariableop_resource:	�Q
Bnews_encoder_batch_normalization_batchnorm_readvariableop_resource:	�G
3news_encoder_dense_1_matmul_readvariableop_resource:
��C
4news_encoder_dense_1_biasadd_readvariableop_resource:	�Y
Jnews_encoder_batch_normalization_1_assignmovingavg_readvariableop_resource:	�[
Lnews_encoder_batch_normalization_1_assignmovingavg_1_readvariableop_resource:	�W
Hnews_encoder_batch_normalization_1_batchnorm_mul_readvariableop_resource:	�S
Dnews_encoder_batch_normalization_1_batchnorm_readvariableop_resource:	�G
3news_encoder_dense_2_matmul_readvariableop_resource:
��C
4news_encoder_dense_2_biasadd_readvariableop_resource:	�Y
Jnews_encoder_batch_normalization_2_assignmovingavg_readvariableop_resource:	�[
Lnews_encoder_batch_normalization_2_assignmovingavg_1_readvariableop_resource:	�W
Hnews_encoder_batch_normalization_2_batchnorm_mul_readvariableop_resource:	�S
Dnews_encoder_batch_normalization_2_batchnorm_readvariableop_resource:	�G
3news_encoder_dense_3_matmul_readvariableop_resource:
��C
4news_encoder_dense_3_biasadd_readvariableop_resource:	�
identity��0news_encoder/batch_normalization/AssignMovingAvg�?news_encoder/batch_normalization/AssignMovingAvg/ReadVariableOp�2news_encoder/batch_normalization/AssignMovingAvg_1�Anews_encoder/batch_normalization/AssignMovingAvg_1/ReadVariableOp�9news_encoder/batch_normalization/batchnorm/ReadVariableOp�=news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp�2news_encoder/batch_normalization_1/AssignMovingAvg�Anews_encoder/batch_normalization_1/AssignMovingAvg/ReadVariableOp�4news_encoder/batch_normalization_1/AssignMovingAvg_1�Cnews_encoder/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�;news_encoder/batch_normalization_1/batchnorm/ReadVariableOp�?news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp�2news_encoder/batch_normalization_2/AssignMovingAvg�Anews_encoder/batch_normalization_2/AssignMovingAvg/ReadVariableOp�4news_encoder/batch_normalization_2/AssignMovingAvg_1�Cnews_encoder/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�;news_encoder/batch_normalization_2/batchnorm/ReadVariableOp�?news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp�)news_encoder/dense/BiasAdd/ReadVariableOp�(news_encoder/dense/MatMul/ReadVariableOp�+news_encoder/dense_1/BiasAdd/ReadVariableOp�*news_encoder/dense_1/MatMul/ReadVariableOp�+news_encoder/dense_2/BiasAdd/ReadVariableOp�*news_encoder/dense_2/MatMul/ReadVariableOp�+news_encoder/dense_3/BiasAdd/ReadVariableOp�*news_encoder/dense_3/MatMul/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
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
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:�����������
(news_encoder/dense/MatMul/ReadVariableOpReadVariableOp1news_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
news_encoder/dense/MatMulMatMulReshape:output:00news_encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
)news_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp2news_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
news_encoder/dense/BiasAddBiasAdd#news_encoder/dense/MatMul:product:01news_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������w
news_encoder/dense/ReluRelu#news_encoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
?news_encoder/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
-news_encoder/batch_normalization/moments/meanMean%news_encoder/dense/Relu:activations:0Hnews_encoder/batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
5news_encoder/batch_normalization/moments/StopGradientStopGradient6news_encoder/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	��
:news_encoder/batch_normalization/moments/SquaredDifferenceSquaredDifference%news_encoder/dense/Relu:activations:0>news_encoder/batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
Cnews_encoder/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
1news_encoder/batch_normalization/moments/varianceMean>news_encoder/batch_normalization/moments/SquaredDifference:z:0Lnews_encoder/batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
0news_encoder/batch_normalization/moments/SqueezeSqueeze6news_encoder/batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
2news_encoder/batch_normalization/moments/Squeeze_1Squeeze:news_encoder/batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 {
6news_encoder/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
?news_encoder/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpHnews_encoder_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4news_encoder/batch_normalization/AssignMovingAvg/subSubGnews_encoder/batch_normalization/AssignMovingAvg/ReadVariableOp:value:09news_encoder/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
4news_encoder/batch_normalization/AssignMovingAvg/mulMul8news_encoder/batch_normalization/AssignMovingAvg/sub:z:0?news_encoder/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
0news_encoder/batch_normalization/AssignMovingAvgAssignSubVariableOpHnews_encoder_batch_normalization_assignmovingavg_readvariableop_resource8news_encoder/batch_normalization/AssignMovingAvg/mul:z:0@^news_encoder/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0}
8news_encoder/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Anews_encoder/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpJnews_encoder_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6news_encoder/batch_normalization/AssignMovingAvg_1/subSubInews_encoder/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0;news_encoder/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
6news_encoder/batch_normalization/AssignMovingAvg_1/mulMul:news_encoder/batch_normalization/AssignMovingAvg_1/sub:z:0Anews_encoder/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization/AssignMovingAvg_1AssignSubVariableOpJnews_encoder_batch_normalization_assignmovingavg_1_readvariableop_resource:news_encoder/batch_normalization/AssignMovingAvg_1/mul:z:0B^news_encoder/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0u
0news_encoder/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
.news_encoder/batch_normalization/batchnorm/addAddV2;news_encoder/batch_normalization/moments/Squeeze_1:output:09news_encoder/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
0news_encoder/batch_normalization/batchnorm/RsqrtRsqrt2news_encoder/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
=news_encoder/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpFnews_encoder_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.news_encoder/batch_normalization/batchnorm/mulMul4news_encoder/batch_normalization/batchnorm/Rsqrt:y:0Enews_encoder/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
0news_encoder/batch_normalization/batchnorm/mul_1Mul%news_encoder/dense/Relu:activations:02news_encoder/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
0news_encoder/batch_normalization/batchnorm/mul_2Mul9news_encoder/batch_normalization/moments/Squeeze:output:02news_encoder/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
9news_encoder/batch_normalization/batchnorm/ReadVariableOpReadVariableOpBnews_encoder_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.news_encoder/batch_normalization/batchnorm/subSubAnews_encoder/batch_normalization/batchnorm/ReadVariableOp:value:04news_encoder/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
0news_encoder/batch_normalization/batchnorm/add_1AddV24news_encoder/batch_normalization/batchnorm/mul_1:z:02news_encoder/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������g
"news_encoder/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
 news_encoder/dropout/dropout/MulMul4news_encoder/batch_normalization/batchnorm/add_1:z:0+news_encoder/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
"news_encoder/dropout/dropout/ShapeShape4news_encoder/batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
9news_encoder/dropout/dropout/random_uniform/RandomUniformRandomUniform+news_encoder/dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*p
+news_encoder/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
)news_encoder/dropout/dropout/GreaterEqualGreaterEqualBnews_encoder/dropout/dropout/random_uniform/RandomUniform:output:04news_encoder/dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������i
$news_encoder/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
%news_encoder/dropout/dropout/SelectV2SelectV2-news_encoder/dropout/dropout/GreaterEqual:z:0$news_encoder/dropout/dropout/Mul:z:0-news_encoder/dropout/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
*news_encoder/dense_1/MatMul/ReadVariableOpReadVariableOp3news_encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
news_encoder/dense_1/MatMulMatMul.news_encoder/dropout/dropout/SelectV2:output:02news_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+news_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp4news_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
news_encoder/dense_1/BiasAddBiasAdd%news_encoder/dense_1/MatMul:product:03news_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
news_encoder/dense_1/ReluRelu%news_encoder/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Anews_encoder/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
/news_encoder/batch_normalization_1/moments/meanMean'news_encoder/dense_1/Relu:activations:0Jnews_encoder/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
7news_encoder/batch_normalization_1/moments/StopGradientStopGradient8news_encoder/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	��
<news_encoder/batch_normalization_1/moments/SquaredDifferenceSquaredDifference'news_encoder/dense_1/Relu:activations:0@news_encoder/batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
Enews_encoder/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
3news_encoder/batch_normalization_1/moments/varianceMean@news_encoder/batch_normalization_1/moments/SquaredDifference:z:0Nnews_encoder/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
2news_encoder/batch_normalization_1/moments/SqueezeSqueeze8news_encoder/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
4news_encoder/batch_normalization_1/moments/Squeeze_1Squeeze<news_encoder/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 }
8news_encoder/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Anews_encoder/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpJnews_encoder_batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6news_encoder/batch_normalization_1/AssignMovingAvg/subSubInews_encoder/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0;news_encoder/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
6news_encoder/batch_normalization_1/AssignMovingAvg/mulMul:news_encoder/batch_normalization_1/AssignMovingAvg/sub:z:0Anews_encoder/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_1/AssignMovingAvgAssignSubVariableOpJnews_encoder_batch_normalization_1_assignmovingavg_readvariableop_resource:news_encoder/batch_normalization_1/AssignMovingAvg/mul:z:0B^news_encoder/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0
:news_encoder/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Cnews_encoder/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpLnews_encoder_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8news_encoder/batch_normalization_1/AssignMovingAvg_1/subSubKnews_encoder/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0=news_encoder/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
8news_encoder/batch_normalization_1/AssignMovingAvg_1/mulMul<news_encoder/batch_normalization_1/AssignMovingAvg_1/sub:z:0Cnews_encoder/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
4news_encoder/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOpLnews_encoder_batch_normalization_1_assignmovingavg_1_readvariableop_resource<news_encoder/batch_normalization_1/AssignMovingAvg_1/mul:z:0D^news_encoder/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0w
2news_encoder/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
0news_encoder/batch_normalization_1/batchnorm/addAddV2=news_encoder/batch_normalization_1/moments/Squeeze_1:output:0;news_encoder/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_1/batchnorm/RsqrtRsqrt4news_encoder/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
?news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpHnews_encoder_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization_1/batchnorm/mulMul6news_encoder/batch_normalization_1/batchnorm/Rsqrt:y:0Gnews_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_1/batchnorm/mul_1Mul'news_encoder/dense_1/Relu:activations:04news_encoder/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
2news_encoder/batch_normalization_1/batchnorm/mul_2Mul;news_encoder/batch_normalization_1/moments/Squeeze:output:04news_encoder/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
;news_encoder/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpDnews_encoder_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization_1/batchnorm/subSubCnews_encoder/batch_normalization_1/batchnorm/ReadVariableOp:value:06news_encoder/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_1/batchnorm/add_1AddV26news_encoder/batch_normalization_1/batchnorm/mul_1:z:04news_encoder/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������i
$news_encoder/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"news_encoder/dropout_1/dropout/MulMul6news_encoder/batch_normalization_1/batchnorm/add_1:z:0-news_encoder/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
$news_encoder/dropout_1/dropout/ShapeShape6news_encoder/batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
;news_encoder/dropout_1/dropout/random_uniform/RandomUniformRandomUniform-news_encoder/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed2*

seed*r
-news_encoder/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
+news_encoder/dropout_1/dropout/GreaterEqualGreaterEqualDnews_encoder/dropout_1/dropout/random_uniform/RandomUniform:output:06news_encoder/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������k
&news_encoder/dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
'news_encoder/dropout_1/dropout/SelectV2SelectV2/news_encoder/dropout_1/dropout/GreaterEqual:z:0&news_encoder/dropout_1/dropout/Mul:z:0/news_encoder/dropout_1/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
*news_encoder/dense_2/MatMul/ReadVariableOpReadVariableOp3news_encoder_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
news_encoder/dense_2/MatMulMatMul0news_encoder/dropout_1/dropout/SelectV2:output:02news_encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+news_encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp4news_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
news_encoder/dense_2/BiasAddBiasAdd%news_encoder/dense_2/MatMul:product:03news_encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
news_encoder/dense_2/ReluRelu%news_encoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Anews_encoder/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
/news_encoder/batch_normalization_2/moments/meanMean'news_encoder/dense_2/Relu:activations:0Jnews_encoder/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
7news_encoder/batch_normalization_2/moments/StopGradientStopGradient8news_encoder/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	��
<news_encoder/batch_normalization_2/moments/SquaredDifferenceSquaredDifference'news_encoder/dense_2/Relu:activations:0@news_encoder/batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
Enews_encoder/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
3news_encoder/batch_normalization_2/moments/varianceMean@news_encoder/batch_normalization_2/moments/SquaredDifference:z:0Nnews_encoder/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
2news_encoder/batch_normalization_2/moments/SqueezeSqueeze8news_encoder/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
4news_encoder/batch_normalization_2/moments/Squeeze_1Squeeze<news_encoder/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 }
8news_encoder/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Anews_encoder/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpJnews_encoder_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6news_encoder/batch_normalization_2/AssignMovingAvg/subSubInews_encoder/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0;news_encoder/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
6news_encoder/batch_normalization_2/AssignMovingAvg/mulMul:news_encoder/batch_normalization_2/AssignMovingAvg/sub:z:0Anews_encoder/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_2/AssignMovingAvgAssignSubVariableOpJnews_encoder_batch_normalization_2_assignmovingavg_readvariableop_resource:news_encoder/batch_normalization_2/AssignMovingAvg/mul:z:0B^news_encoder/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0
:news_encoder/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Cnews_encoder/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpLnews_encoder_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8news_encoder/batch_normalization_2/AssignMovingAvg_1/subSubKnews_encoder/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0=news_encoder/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
8news_encoder/batch_normalization_2/AssignMovingAvg_1/mulMul<news_encoder/batch_normalization_2/AssignMovingAvg_1/sub:z:0Cnews_encoder/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
4news_encoder/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpLnews_encoder_batch_normalization_2_assignmovingavg_1_readvariableop_resource<news_encoder/batch_normalization_2/AssignMovingAvg_1/mul:z:0D^news_encoder/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0w
2news_encoder/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
0news_encoder/batch_normalization_2/batchnorm/addAddV2=news_encoder/batch_normalization_2/moments/Squeeze_1:output:0;news_encoder/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_2/batchnorm/RsqrtRsqrt4news_encoder/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
?news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpHnews_encoder_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization_2/batchnorm/mulMul6news_encoder/batch_normalization_2/batchnorm/Rsqrt:y:0Gnews_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_2/batchnorm/mul_1Mul'news_encoder/dense_2/Relu:activations:04news_encoder/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
2news_encoder/batch_normalization_2/batchnorm/mul_2Mul;news_encoder/batch_normalization_2/moments/Squeeze:output:04news_encoder/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
;news_encoder/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpDnews_encoder_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0news_encoder/batch_normalization_2/batchnorm/subSubCnews_encoder/batch_normalization_2/batchnorm/ReadVariableOp:value:06news_encoder/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
2news_encoder/batch_normalization_2/batchnorm/add_1AddV26news_encoder/batch_normalization_2/batchnorm/mul_1:z:04news_encoder/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������i
$news_encoder/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"news_encoder/dropout_2/dropout/MulMul6news_encoder/batch_normalization_2/batchnorm/add_1:z:0-news_encoder/dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
$news_encoder/dropout_2/dropout/ShapeShape6news_encoder/batch_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
;news_encoder/dropout_2/dropout/random_uniform/RandomUniformRandomUniform-news_encoder/dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed2*

seed*r
-news_encoder/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
+news_encoder/dropout_2/dropout/GreaterEqualGreaterEqualDnews_encoder/dropout_2/dropout/random_uniform/RandomUniform:output:06news_encoder/dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������k
&news_encoder/dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
'news_encoder/dropout_2/dropout/SelectV2SelectV2/news_encoder/dropout_2/dropout/GreaterEqual:z:0&news_encoder/dropout_2/dropout/Mul:z:0/news_encoder/dropout_2/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
*news_encoder/dense_3/MatMul/ReadVariableOpReadVariableOp3news_encoder_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
news_encoder/dense_3/MatMulMatMul0news_encoder/dropout_2/dropout/SelectV2:output:02news_encoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+news_encoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp4news_encoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
news_encoder/dense_3/BiasAddBiasAdd%news_encoder/dense_3/MatMul:product:03news_encoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
news_encoder/dense_3/ReluRelu%news_encoder/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape'news_encoder/dense_3/Relu:activations:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:��������������������
NoOpNoOp1^news_encoder/batch_normalization/AssignMovingAvg@^news_encoder/batch_normalization/AssignMovingAvg/ReadVariableOp3^news_encoder/batch_normalization/AssignMovingAvg_1B^news_encoder/batch_normalization/AssignMovingAvg_1/ReadVariableOp:^news_encoder/batch_normalization/batchnorm/ReadVariableOp>^news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp3^news_encoder/batch_normalization_1/AssignMovingAvgB^news_encoder/batch_normalization_1/AssignMovingAvg/ReadVariableOp5^news_encoder/batch_normalization_1/AssignMovingAvg_1D^news_encoder/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp<^news_encoder/batch_normalization_1/batchnorm/ReadVariableOp@^news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp3^news_encoder/batch_normalization_2/AssignMovingAvgB^news_encoder/batch_normalization_2/AssignMovingAvg/ReadVariableOp5^news_encoder/batch_normalization_2/AssignMovingAvg_1D^news_encoder/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp<^news_encoder/batch_normalization_2/batchnorm/ReadVariableOp@^news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp*^news_encoder/dense/BiasAdd/ReadVariableOp)^news_encoder/dense/MatMul/ReadVariableOp,^news_encoder/dense_1/BiasAdd/ReadVariableOp+^news_encoder/dense_1/MatMul/ReadVariableOp,^news_encoder/dense_2/BiasAdd/ReadVariableOp+^news_encoder/dense_2/MatMul/ReadVariableOp,^news_encoder/dense_3/BiasAdd/ReadVariableOp+^news_encoder/dense_3/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�������������������: : : : : : : : : : : : : : : : : : : : 2�
?news_encoder/batch_normalization/AssignMovingAvg/ReadVariableOp?news_encoder/batch_normalization/AssignMovingAvg/ReadVariableOp2�
Anews_encoder/batch_normalization/AssignMovingAvg_1/ReadVariableOpAnews_encoder/batch_normalization/AssignMovingAvg_1/ReadVariableOp2h
2news_encoder/batch_normalization/AssignMovingAvg_12news_encoder/batch_normalization/AssignMovingAvg_12d
0news_encoder/batch_normalization/AssignMovingAvg0news_encoder/batch_normalization/AssignMovingAvg2v
9news_encoder/batch_normalization/batchnorm/ReadVariableOp9news_encoder/batch_normalization/batchnorm/ReadVariableOp2~
=news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp=news_encoder/batch_normalization/batchnorm/mul/ReadVariableOp2�
Anews_encoder/batch_normalization_1/AssignMovingAvg/ReadVariableOpAnews_encoder/batch_normalization_1/AssignMovingAvg/ReadVariableOp2�
Cnews_encoder/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpCnews_encoder/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2l
4news_encoder/batch_normalization_1/AssignMovingAvg_14news_encoder/batch_normalization_1/AssignMovingAvg_12h
2news_encoder/batch_normalization_1/AssignMovingAvg2news_encoder/batch_normalization_1/AssignMovingAvg2z
;news_encoder/batch_normalization_1/batchnorm/ReadVariableOp;news_encoder/batch_normalization_1/batchnorm/ReadVariableOp2�
?news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp?news_encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp2�
Anews_encoder/batch_normalization_2/AssignMovingAvg/ReadVariableOpAnews_encoder/batch_normalization_2/AssignMovingAvg/ReadVariableOp2�
Cnews_encoder/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpCnews_encoder/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2l
4news_encoder/batch_normalization_2/AssignMovingAvg_14news_encoder/batch_normalization_2/AssignMovingAvg_12h
2news_encoder/batch_normalization_2/AssignMovingAvg2news_encoder/batch_normalization_2/AssignMovingAvg2z
;news_encoder/batch_normalization_2/batchnorm/ReadVariableOp;news_encoder/batch_normalization_2/batchnorm/ReadVariableOp2�
?news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp?news_encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp2V
)news_encoder/dense/BiasAdd/ReadVariableOp)news_encoder/dense/BiasAdd/ReadVariableOp2T
(news_encoder/dense/MatMul/ReadVariableOp(news_encoder/dense/MatMul/ReadVariableOp2Z
+news_encoder/dense_1/BiasAdd/ReadVariableOp+news_encoder/dense_1/BiasAdd/ReadVariableOp2X
*news_encoder/dense_1/MatMul/ReadVariableOp*news_encoder/dense_1/MatMul/ReadVariableOp2Z
+news_encoder/dense_2/BiasAdd/ReadVariableOp+news_encoder/dense_2/BiasAdd/ReadVariableOp2X
*news_encoder/dense_2/MatMul/ReadVariableOp*news_encoder/dense_2/MatMul/ReadVariableOp2Z
+news_encoder/dense_3/BiasAdd/ReadVariableOp+news_encoder/dense_3/BiasAdd/ReadVariableOp2X
*news_encoder/dense_3/MatMul/ReadVariableOp*news_encoder/dense_3/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
%__inference_dense_layer_call_fn_86211

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_83201p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name86207:%!

_user_specified_name86205:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_news_encoder_layer_call_fn_83436
input_4
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
��

unknown_18:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*0
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *P
fKRI
G__inference_news_encoder_layer_call_and_return_conditional_losses_83322p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name83432:%!

_user_specified_name83430:%!

_user_specified_name83428:%!

_user_specified_name83426:%!

_user_specified_name83424:%!

_user_specified_name83422:%!

_user_specified_name83420:%!

_user_specified_name83418:%!

_user_specified_name83416:%!

_user_specified_name83414:%
!

_user_specified_name83412:%	!

_user_specified_name83410:%!

_user_specified_name83408:%!

_user_specified_name83406:%!

_user_specified_name83404:%!

_user_specified_name83402:%!

_user_specified_name83400:%!

_user_specified_name83398:%!

_user_specified_name83396:%!

_user_specified_name83394:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_4
�	
�
3__inference_batch_normalization_layer_call_fn_86235

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_82982p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name86231:%!

_user_specified_name86229:%!

_user_specified_name86227:%!

_user_specified_name86225:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
ծ
�
@__inference_model_layer_call_and_return_conditional_losses_84876
input_1
input_2,
time_distributed_1_84660:
��'
time_distributed_1_84662:	�'
time_distributed_1_84664:	�'
time_distributed_1_84666:	�'
time_distributed_1_84668:	�'
time_distributed_1_84670:	�,
time_distributed_1_84672:
��'
time_distributed_1_84674:	�'
time_distributed_1_84676:	�'
time_distributed_1_84678:	�'
time_distributed_1_84680:	�'
time_distributed_1_84682:	�,
time_distributed_1_84684:
��'
time_distributed_1_84686:	�'
time_distributed_1_84688:	�'
time_distributed_1_84690:	�'
time_distributed_1_84692:	�'
time_distributed_1_84694:	�,
time_distributed_1_84696:
��'
time_distributed_1_84698:	�&
user_encoder_84845:
��&
user_encoder_84847:
��&
user_encoder_84849:
��&
user_encoder_84851:
��!
user_encoder_84853:	�%
user_encoder_84855:	�
identity��*time_distributed_1/StatefulPartitionedCall�6time_distributed_1/batch_normalization/AssignMovingAvg�Etime_distributed_1/batch_normalization/AssignMovingAvg/ReadVariableOp�8time_distributed_1/batch_normalization/AssignMovingAvg_1�Gtime_distributed_1/batch_normalization/AssignMovingAvg_1/ReadVariableOp�?time_distributed_1/batch_normalization/batchnorm/ReadVariableOp�Ctime_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOp�8time_distributed_1/batch_normalization_1/AssignMovingAvg�Gtime_distributed_1/batch_normalization_1/AssignMovingAvg/ReadVariableOp�:time_distributed_1/batch_normalization_1/AssignMovingAvg_1�Itime_distributed_1/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�Atime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp�Etime_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOp�8time_distributed_1/batch_normalization_2/AssignMovingAvg�Gtime_distributed_1/batch_normalization_2/AssignMovingAvg/ReadVariableOp�:time_distributed_1/batch_normalization_2/AssignMovingAvg_1�Itime_distributed_1/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�Atime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp�Etime_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOp�/time_distributed_1/dense/BiasAdd/ReadVariableOp�.time_distributed_1/dense/MatMul/ReadVariableOp�1time_distributed_1/dense_1/BiasAdd/ReadVariableOp�0time_distributed_1/dense_1/MatMul/ReadVariableOp�1time_distributed_1/dense_2/BiasAdd/ReadVariableOp�0time_distributed_1/dense_2/MatMul/ReadVariableOp�1time_distributed_1/dense_3/BiasAdd/ReadVariableOp�0time_distributed_1/dense_3/MatMul/ReadVariableOp�$user_encoder/StatefulPartitionedCall�
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCallinput_2time_distributed_1_84660time_distributed_1_84662time_distributed_1_84664time_distributed_1_84666time_distributed_1_84668time_distributed_1_84670time_distributed_1_84672time_distributed_1_84674time_distributed_1_84676time_distributed_1_84678time_distributed_1_84680time_distributed_1_84682time_distributed_1_84684time_distributed_1_84686time_distributed_1_84688time_distributed_1_84690time_distributed_1_84692time_distributed_1_84694time_distributed_1_84696time_distributed_1_84698* 
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*0
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_83603q
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
time_distributed_1/ReshapeReshapeinput_2)time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
.time_distributed_1/dense/MatMul/ReadVariableOpReadVariableOptime_distributed_1_84660* 
_output_shapes
:
��*
dtype0�
time_distributed_1/dense/MatMulMatMul#time_distributed_1/Reshape:output:06time_distributed_1/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/time_distributed_1/dense/BiasAdd/ReadVariableOpReadVariableOptime_distributed_1_84662*
_output_shapes	
:�*
dtype0�
 time_distributed_1/dense/BiasAddBiasAdd)time_distributed_1/dense/MatMul:product:07time_distributed_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
time_distributed_1/dense/ReluRelu)time_distributed_1/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Etime_distributed_1/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
3time_distributed_1/batch_normalization/moments/meanMean+time_distributed_1/dense/Relu:activations:0Ntime_distributed_1/batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
;time_distributed_1/batch_normalization/moments/StopGradientStopGradient<time_distributed_1/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	��
@time_distributed_1/batch_normalization/moments/SquaredDifferenceSquaredDifference+time_distributed_1/dense/Relu:activations:0Dtime_distributed_1/batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
Itime_distributed_1/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
7time_distributed_1/batch_normalization/moments/varianceMeanDtime_distributed_1/batch_normalization/moments/SquaredDifference:z:0Rtime_distributed_1/batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
6time_distributed_1/batch_normalization/moments/SqueezeSqueeze<time_distributed_1/batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
8time_distributed_1/batch_normalization/moments/Squeeze_1Squeeze@time_distributed_1/batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
<time_distributed_1/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Etime_distributed_1/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOptime_distributed_1_84664+^time_distributed_1/StatefulPartitionedCall*
_output_shapes	
:�*
dtype0�
:time_distributed_1/batch_normalization/AssignMovingAvg/subSubMtime_distributed_1/batch_normalization/AssignMovingAvg/ReadVariableOp:value:0?time_distributed_1/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
:time_distributed_1/batch_normalization/AssignMovingAvg/mulMul>time_distributed_1/batch_normalization/AssignMovingAvg/sub:z:0Etime_distributed_1/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
6time_distributed_1/batch_normalization/AssignMovingAvgAssignSubVariableOptime_distributed_1_84664>time_distributed_1/batch_normalization/AssignMovingAvg/mul:z:0+^time_distributed_1/StatefulPartitionedCallF^time_distributed_1/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0�
>time_distributed_1/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Gtime_distributed_1/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOptime_distributed_1_84666+^time_distributed_1/StatefulPartitionedCall*
_output_shapes	
:�*
dtype0�
<time_distributed_1/batch_normalization/AssignMovingAvg_1/subSubOtime_distributed_1/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0Atime_distributed_1/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
<time_distributed_1/batch_normalization/AssignMovingAvg_1/mulMul@time_distributed_1/batch_normalization/AssignMovingAvg_1/sub:z:0Gtime_distributed_1/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
8time_distributed_1/batch_normalization/AssignMovingAvg_1AssignSubVariableOptime_distributed_1_84666@time_distributed_1/batch_normalization/AssignMovingAvg_1/mul:z:0+^time_distributed_1/StatefulPartitionedCallH^time_distributed_1/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0{
6time_distributed_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4time_distributed_1/batch_normalization/batchnorm/addAddV2Atime_distributed_1/batch_normalization/moments/Squeeze_1:output:0?time_distributed_1/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
6time_distributed_1/batch_normalization/batchnorm/RsqrtRsqrt8time_distributed_1/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Ctime_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOptime_distributed_1_84668*
_output_shapes	
:�*
dtype0�
4time_distributed_1/batch_normalization/batchnorm/mulMul:time_distributed_1/batch_normalization/batchnorm/Rsqrt:y:0Ktime_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
6time_distributed_1/batch_normalization/batchnorm/mul_1Mul+time_distributed_1/dense/Relu:activations:08time_distributed_1/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
6time_distributed_1/batch_normalization/batchnorm/mul_2Mul?time_distributed_1/batch_normalization/moments/Squeeze:output:08time_distributed_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
?time_distributed_1/batch_normalization/batchnorm/ReadVariableOpReadVariableOptime_distributed_1_84670*
_output_shapes	
:�*
dtype0�
4time_distributed_1/batch_normalization/batchnorm/subSubGtime_distributed_1/batch_normalization/batchnorm/ReadVariableOp:value:0:time_distributed_1/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
6time_distributed_1/batch_normalization/batchnorm/add_1AddV2:time_distributed_1/batch_normalization/batchnorm/mul_1:z:08time_distributed_1/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������m
(time_distributed_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
&time_distributed_1/dropout/dropout/MulMul:time_distributed_1/batch_normalization/batchnorm/add_1:z:01time_distributed_1/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
(time_distributed_1/dropout/dropout/ShapeShape:time_distributed_1/batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
?time_distributed_1/dropout/dropout/random_uniform/RandomUniformRandomUniform1time_distributed_1/dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*v
1time_distributed_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
/time_distributed_1/dropout/dropout/GreaterEqualGreaterEqualHtime_distributed_1/dropout/dropout/random_uniform/RandomUniform:output:0:time_distributed_1/dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������o
*time_distributed_1/dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
+time_distributed_1/dropout/dropout/SelectV2SelectV23time_distributed_1/dropout/dropout/GreaterEqual:z:0*time_distributed_1/dropout/dropout/Mul:z:03time_distributed_1/dropout/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
0time_distributed_1/dense_1/MatMul/ReadVariableOpReadVariableOptime_distributed_1_84672* 
_output_shapes
:
��*
dtype0�
!time_distributed_1/dense_1/MatMulMatMul4time_distributed_1/dropout/dropout/SelectV2:output:08time_distributed_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
1time_distributed_1/dense_1/BiasAdd/ReadVariableOpReadVariableOptime_distributed_1_84674*
_output_shapes	
:�*
dtype0�
"time_distributed_1/dense_1/BiasAddBiasAdd+time_distributed_1/dense_1/MatMul:product:09time_distributed_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
time_distributed_1/dense_1/ReluRelu+time_distributed_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Gtime_distributed_1/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
5time_distributed_1/batch_normalization_1/moments/meanMean-time_distributed_1/dense_1/Relu:activations:0Ptime_distributed_1/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
=time_distributed_1/batch_normalization_1/moments/StopGradientStopGradient>time_distributed_1/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	��
Btime_distributed_1/batch_normalization_1/moments/SquaredDifferenceSquaredDifference-time_distributed_1/dense_1/Relu:activations:0Ftime_distributed_1/batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
Ktime_distributed_1/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
9time_distributed_1/batch_normalization_1/moments/varianceMeanFtime_distributed_1/batch_normalization_1/moments/SquaredDifference:z:0Ttime_distributed_1/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
8time_distributed_1/batch_normalization_1/moments/SqueezeSqueeze>time_distributed_1/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
:time_distributed_1/batch_normalization_1/moments/Squeeze_1SqueezeBtime_distributed_1/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
>time_distributed_1/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Gtime_distributed_1/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOptime_distributed_1_84676+^time_distributed_1/StatefulPartitionedCall*
_output_shapes	
:�*
dtype0�
<time_distributed_1/batch_normalization_1/AssignMovingAvg/subSubOtime_distributed_1/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0Atime_distributed_1/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
<time_distributed_1/batch_normalization_1/AssignMovingAvg/mulMul@time_distributed_1/batch_normalization_1/AssignMovingAvg/sub:z:0Gtime_distributed_1/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
8time_distributed_1/batch_normalization_1/AssignMovingAvgAssignSubVariableOptime_distributed_1_84676@time_distributed_1/batch_normalization_1/AssignMovingAvg/mul:z:0+^time_distributed_1/StatefulPartitionedCallH^time_distributed_1/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0�
@time_distributed_1/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Itime_distributed_1/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOptime_distributed_1_84678+^time_distributed_1/StatefulPartitionedCall*
_output_shapes	
:�*
dtype0�
>time_distributed_1/batch_normalization_1/AssignMovingAvg_1/subSubQtime_distributed_1/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0Ctime_distributed_1/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
>time_distributed_1/batch_normalization_1/AssignMovingAvg_1/mulMulBtime_distributed_1/batch_normalization_1/AssignMovingAvg_1/sub:z:0Itime_distributed_1/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
:time_distributed_1/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOptime_distributed_1_84678Btime_distributed_1/batch_normalization_1/AssignMovingAvg_1/mul:z:0+^time_distributed_1/StatefulPartitionedCallJ^time_distributed_1/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0}
8time_distributed_1/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
6time_distributed_1/batch_normalization_1/batchnorm/addAddV2Ctime_distributed_1/batch_normalization_1/moments/Squeeze_1:output:0Atime_distributed_1/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
8time_distributed_1/batch_normalization_1/batchnorm/RsqrtRsqrt:time_distributed_1/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Etime_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOptime_distributed_1_84680*
_output_shapes	
:�*
dtype0�
6time_distributed_1/batch_normalization_1/batchnorm/mulMul<time_distributed_1/batch_normalization_1/batchnorm/Rsqrt:y:0Mtime_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
8time_distributed_1/batch_normalization_1/batchnorm/mul_1Mul-time_distributed_1/dense_1/Relu:activations:0:time_distributed_1/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
8time_distributed_1/batch_normalization_1/batchnorm/mul_2MulAtime_distributed_1/batch_normalization_1/moments/Squeeze:output:0:time_distributed_1/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
Atime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOptime_distributed_1_84682*
_output_shapes	
:�*
dtype0�
6time_distributed_1/batch_normalization_1/batchnorm/subSubItime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp:value:0<time_distributed_1/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
8time_distributed_1/batch_normalization_1/batchnorm/add_1AddV2<time_distributed_1/batch_normalization_1/batchnorm/mul_1:z:0:time_distributed_1/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������o
*time_distributed_1/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(time_distributed_1/dropout_1/dropout/MulMul<time_distributed_1/batch_normalization_1/batchnorm/add_1:z:03time_distributed_1/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
*time_distributed_1/dropout_1/dropout/ShapeShape<time_distributed_1/batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
Atime_distributed_1/dropout_1/dropout/random_uniform/RandomUniformRandomUniform3time_distributed_1/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed2*

seed*x
3time_distributed_1/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
1time_distributed_1/dropout_1/dropout/GreaterEqualGreaterEqualJtime_distributed_1/dropout_1/dropout/random_uniform/RandomUniform:output:0<time_distributed_1/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������q
,time_distributed_1/dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
-time_distributed_1/dropout_1/dropout/SelectV2SelectV25time_distributed_1/dropout_1/dropout/GreaterEqual:z:0,time_distributed_1/dropout_1/dropout/Mul:z:05time_distributed_1/dropout_1/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
0time_distributed_1/dense_2/MatMul/ReadVariableOpReadVariableOptime_distributed_1_84684* 
_output_shapes
:
��*
dtype0�
!time_distributed_1/dense_2/MatMulMatMul6time_distributed_1/dropout_1/dropout/SelectV2:output:08time_distributed_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
1time_distributed_1/dense_2/BiasAdd/ReadVariableOpReadVariableOptime_distributed_1_84686*
_output_shapes	
:�*
dtype0�
"time_distributed_1/dense_2/BiasAddBiasAdd+time_distributed_1/dense_2/MatMul:product:09time_distributed_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
time_distributed_1/dense_2/ReluRelu+time_distributed_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Gtime_distributed_1/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
5time_distributed_1/batch_normalization_2/moments/meanMean-time_distributed_1/dense_2/Relu:activations:0Ptime_distributed_1/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
=time_distributed_1/batch_normalization_2/moments/StopGradientStopGradient>time_distributed_1/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	��
Btime_distributed_1/batch_normalization_2/moments/SquaredDifferenceSquaredDifference-time_distributed_1/dense_2/Relu:activations:0Ftime_distributed_1/batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
Ktime_distributed_1/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
9time_distributed_1/batch_normalization_2/moments/varianceMeanFtime_distributed_1/batch_normalization_2/moments/SquaredDifference:z:0Ttime_distributed_1/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
8time_distributed_1/batch_normalization_2/moments/SqueezeSqueeze>time_distributed_1/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
:time_distributed_1/batch_normalization_2/moments/Squeeze_1SqueezeBtime_distributed_1/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
>time_distributed_1/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Gtime_distributed_1/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOptime_distributed_1_84688+^time_distributed_1/StatefulPartitionedCall*
_output_shapes	
:�*
dtype0�
<time_distributed_1/batch_normalization_2/AssignMovingAvg/subSubOtime_distributed_1/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0Atime_distributed_1/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
<time_distributed_1/batch_normalization_2/AssignMovingAvg/mulMul@time_distributed_1/batch_normalization_2/AssignMovingAvg/sub:z:0Gtime_distributed_1/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
8time_distributed_1/batch_normalization_2/AssignMovingAvgAssignSubVariableOptime_distributed_1_84688@time_distributed_1/batch_normalization_2/AssignMovingAvg/mul:z:0+^time_distributed_1/StatefulPartitionedCallH^time_distributed_1/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0�
@time_distributed_1/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Itime_distributed_1/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOptime_distributed_1_84690+^time_distributed_1/StatefulPartitionedCall*
_output_shapes	
:�*
dtype0�
>time_distributed_1/batch_normalization_2/AssignMovingAvg_1/subSubQtime_distributed_1/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0Ctime_distributed_1/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
>time_distributed_1/batch_normalization_2/AssignMovingAvg_1/mulMulBtime_distributed_1/batch_normalization_2/AssignMovingAvg_1/sub:z:0Itime_distributed_1/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
:time_distributed_1/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOptime_distributed_1_84690Btime_distributed_1/batch_normalization_2/AssignMovingAvg_1/mul:z:0+^time_distributed_1/StatefulPartitionedCallJ^time_distributed_1/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0}
8time_distributed_1/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
6time_distributed_1/batch_normalization_2/batchnorm/addAddV2Ctime_distributed_1/batch_normalization_2/moments/Squeeze_1:output:0Atime_distributed_1/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
8time_distributed_1/batch_normalization_2/batchnorm/RsqrtRsqrt:time_distributed_1/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
Etime_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOptime_distributed_1_84692*
_output_shapes	
:�*
dtype0�
6time_distributed_1/batch_normalization_2/batchnorm/mulMul<time_distributed_1/batch_normalization_2/batchnorm/Rsqrt:y:0Mtime_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
8time_distributed_1/batch_normalization_2/batchnorm/mul_1Mul-time_distributed_1/dense_2/Relu:activations:0:time_distributed_1/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
8time_distributed_1/batch_normalization_2/batchnorm/mul_2MulAtime_distributed_1/batch_normalization_2/moments/Squeeze:output:0:time_distributed_1/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
Atime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOptime_distributed_1_84694*
_output_shapes	
:�*
dtype0�
6time_distributed_1/batch_normalization_2/batchnorm/subSubItime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp:value:0<time_distributed_1/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
8time_distributed_1/batch_normalization_2/batchnorm/add_1AddV2<time_distributed_1/batch_normalization_2/batchnorm/mul_1:z:0:time_distributed_1/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������o
*time_distributed_1/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(time_distributed_1/dropout_2/dropout/MulMul<time_distributed_1/batch_normalization_2/batchnorm/add_1:z:03time_distributed_1/dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
*time_distributed_1/dropout_2/dropout/ShapeShape<time_distributed_1/batch_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
Atime_distributed_1/dropout_2/dropout/random_uniform/RandomUniformRandomUniform3time_distributed_1/dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed2*

seed*x
3time_distributed_1/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
1time_distributed_1/dropout_2/dropout/GreaterEqualGreaterEqualJtime_distributed_1/dropout_2/dropout/random_uniform/RandomUniform:output:0<time_distributed_1/dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������q
,time_distributed_1/dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
-time_distributed_1/dropout_2/dropout/SelectV2SelectV25time_distributed_1/dropout_2/dropout/GreaterEqual:z:0,time_distributed_1/dropout_2/dropout/Mul:z:05time_distributed_1/dropout_2/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
0time_distributed_1/dense_3/MatMul/ReadVariableOpReadVariableOptime_distributed_1_84696* 
_output_shapes
:
��*
dtype0�
!time_distributed_1/dense_3/MatMulMatMul6time_distributed_1/dropout_2/dropout/SelectV2:output:08time_distributed_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
1time_distributed_1/dense_3/BiasAdd/ReadVariableOpReadVariableOptime_distributed_1_84698*
_output_shapes	
:�*
dtype0�
"time_distributed_1/dense_3/BiasAddBiasAdd+time_distributed_1/dense_3/MatMul:product:09time_distributed_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
time_distributed_1/dense_3/ReluRelu+time_distributed_1/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������

$user_encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1time_distributed_1_84660time_distributed_1_84662time_distributed_1_84664time_distributed_1_84666time_distributed_1_84668time_distributed_1_84670time_distributed_1_84672time_distributed_1_84674time_distributed_1_84676time_distributed_1_84678time_distributed_1_84680time_distributed_1_84682time_distributed_1_84684time_distributed_1_84686time_distributed_1_84688time_distributed_1_84690time_distributed_1_84692time_distributed_1_84694time_distributed_1_84696time_distributed_1_84698user_encoder_84845user_encoder_84847user_encoder_84849user_encoder_84851user_encoder_84853user_encoder_848557^time_distributed_1/batch_normalization/AssignMovingAvg9^time_distributed_1/batch_normalization/AssignMovingAvg_19^time_distributed_1/batch_normalization_1/AssignMovingAvg;^time_distributed_1/batch_normalization_1/AssignMovingAvg_19^time_distributed_1/batch_normalization_2/AssignMovingAvg;^time_distributed_1/batch_normalization_2/AssignMovingAvg_1*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*6
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU 2J 8R(���������� *P
fKRI
G__inference_user_encoder_layer_call_and_return_conditional_losses_84325�
dot/PartitionedCallPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0-user_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *G
fBR@
>__inference_dot_layer_call_and_return_conditional_losses_84867�
activation/PartitionedCallPartitionedCalldot/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU 2J 8R(���������� *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_84873{
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:�������������������
NoOpNoOp+^time_distributed_1/StatefulPartitionedCall7^time_distributed_1/batch_normalization/AssignMovingAvgF^time_distributed_1/batch_normalization/AssignMovingAvg/ReadVariableOp9^time_distributed_1/batch_normalization/AssignMovingAvg_1H^time_distributed_1/batch_normalization/AssignMovingAvg_1/ReadVariableOp@^time_distributed_1/batch_normalization/batchnorm/ReadVariableOpD^time_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOp9^time_distributed_1/batch_normalization_1/AssignMovingAvgH^time_distributed_1/batch_normalization_1/AssignMovingAvg/ReadVariableOp;^time_distributed_1/batch_normalization_1/AssignMovingAvg_1J^time_distributed_1/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpB^time_distributed_1/batch_normalization_1/batchnorm/ReadVariableOpF^time_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOp9^time_distributed_1/batch_normalization_2/AssignMovingAvgH^time_distributed_1/batch_normalization_2/AssignMovingAvg/ReadVariableOp;^time_distributed_1/batch_normalization_2/AssignMovingAvg_1J^time_distributed_1/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpB^time_distributed_1/batch_normalization_2/batchnorm/ReadVariableOpF^time_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOp0^time_distributed_1/dense/BiasAdd/ReadVariableOp/^time_distributed_1/dense/MatMul/ReadVariableOp2^time_distributed_1/dense_1/BiasAdd/ReadVariableOp1^time_distributed_1/dense_1/MatMul/ReadVariableOp2^time_distributed_1/dense_2/BiasAdd/ReadVariableOp1^time_distributed_1/dense_2/MatMul/ReadVariableOp2^time_distributed_1/dense_3/BiasAdd/ReadVariableOp1^time_distributed_1/dense_3/MatMul/ReadVariableOp%^user_encoder/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:���������#�:�������������������: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall2�
Etime_distributed_1/batch_normalization/AssignMovingAvg/ReadVariableOpEtime_distributed_1/batch_normalization/AssignMovingAvg/ReadVariableOp2�
Gtime_distributed_1/batch_normalization/AssignMovingAvg_1/ReadVariableOpGtime_distributed_1/batch_normalization/AssignMovingAvg_1/ReadVariableOp2t
8time_distributed_1/batch_normalization/AssignMovingAvg_18time_distributed_1/batch_normalization/AssignMovingAvg_12p
6time_distributed_1/batch_normalization/AssignMovingAvg6time_distributed_1/batch_normalization/AssignMovingAvg2�
?time_distributed_1/batch_normalization/batchnorm/ReadVariableOp?time_distributed_1/batch_normalization/batchnorm/ReadVariableOp2�
Ctime_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOpCtime_distributed_1/batch_normalization/batchnorm/mul/ReadVariableOp2�
Gtime_distributed_1/batch_normalization_1/AssignMovingAvg/ReadVariableOpGtime_distributed_1/batch_normalization_1/AssignMovingAvg/ReadVariableOp2�
Itime_distributed_1/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpItime_distributed_1/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2x
:time_distributed_1/batch_normalization_1/AssignMovingAvg_1:time_distributed_1/batch_normalization_1/AssignMovingAvg_12t
8time_distributed_1/batch_normalization_1/AssignMovingAvg8time_distributed_1/batch_normalization_1/AssignMovingAvg2�
Atime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOpAtime_distributed_1/batch_normalization_1/batchnorm/ReadVariableOp2�
Etime_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOpEtime_distributed_1/batch_normalization_1/batchnorm/mul/ReadVariableOp2�
Gtime_distributed_1/batch_normalization_2/AssignMovingAvg/ReadVariableOpGtime_distributed_1/batch_normalization_2/AssignMovingAvg/ReadVariableOp2�
Itime_distributed_1/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpItime_distributed_1/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2x
:time_distributed_1/batch_normalization_2/AssignMovingAvg_1:time_distributed_1/batch_normalization_2/AssignMovingAvg_12t
8time_distributed_1/batch_normalization_2/AssignMovingAvg8time_distributed_1/batch_normalization_2/AssignMovingAvg2�
Atime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOpAtime_distributed_1/batch_normalization_2/batchnorm/ReadVariableOp2�
Etime_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOpEtime_distributed_1/batch_normalization_2/batchnorm/mul/ReadVariableOp2b
/time_distributed_1/dense/BiasAdd/ReadVariableOp/time_distributed_1/dense/BiasAdd/ReadVariableOp2`
.time_distributed_1/dense/MatMul/ReadVariableOp.time_distributed_1/dense/MatMul/ReadVariableOp2f
1time_distributed_1/dense_1/BiasAdd/ReadVariableOp1time_distributed_1/dense_1/BiasAdd/ReadVariableOp2d
0time_distributed_1/dense_1/MatMul/ReadVariableOp0time_distributed_1/dense_1/MatMul/ReadVariableOp2f
1time_distributed_1/dense_2/BiasAdd/ReadVariableOp1time_distributed_1/dense_2/BiasAdd/ReadVariableOp2d
0time_distributed_1/dense_2/MatMul/ReadVariableOp0time_distributed_1/dense_2/MatMul/ReadVariableOp2f
1time_distributed_1/dense_3/BiasAdd/ReadVariableOp1time_distributed_1/dense_3/BiasAdd/ReadVariableOp2d
0time_distributed_1/dense_3/MatMul/ReadVariableOp0time_distributed_1/dense_3/MatMul/ReadVariableOp2L
$user_encoder/StatefulPartitionedCall$user_encoder/StatefulPartitionedCall:%!

_user_specified_name84855:%!

_user_specified_name84853:%!

_user_specified_name84851:%!

_user_specified_name84849:%!

_user_specified_name84847:%!

_user_specified_name84845:%!

_user_specified_name84698:%!

_user_specified_name84696:%!

_user_specified_name84694:%!

_user_specified_name84692:%!

_user_specified_name84690:%!

_user_specified_name84688:%!

_user_specified_name84686:%!

_user_specified_name84684:%!

_user_specified_name84682:%!

_user_specified_name84680:%!

_user_specified_name84678:%
!

_user_specified_name84676:%	!

_user_specified_name84674:%!

_user_specified_name84672:%!

_user_specified_name84670:%!

_user_specified_name84668:%!

_user_specified_name84666:%!

_user_specified_name84664:%!

_user_specified_name84662:%!

_user_specified_name84660:^Z
5
_output_shapes#
!:�������������������
!
_user_specified_name	input_2:U Q
,
_output_shapes
:���������#�
!
_user_specified_name	input_1
�
h
>__inference_dot_layer_call_and_return_conditional_losses_84867

inputs
inputs_1
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :r

ExpandDims
ExpandDimsinputs_1ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������s
MatMulBatchMatMulV2inputsExpandDims:output:0*
T0*4
_output_shapes"
 :������������������R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::��~
SqueezeSqueezeMatMul:output:0*
T0*0
_output_shapes
:������������������*
squeeze_dims

���������a
IdentityIdentitySqueeze:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:�������������������:����������:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�

�
@__inference_dense_layer_call_and_return_conditional_losses_83201

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�!
�
K__inference_time_distributed_layer_call_and_return_conditional_losses_83861

inputs&
news_encoder_83815:
��!
news_encoder_83817:	�!
news_encoder_83819:	�!
news_encoder_83821:	�!
news_encoder_83823:	�!
news_encoder_83825:	�&
news_encoder_83827:
��!
news_encoder_83829:	�!
news_encoder_83831:	�!
news_encoder_83833:	�!
news_encoder_83835:	�!
news_encoder_83837:	�&
news_encoder_83839:
��!
news_encoder_83841:	�!
news_encoder_83843:	�!
news_encoder_83845:	�!
news_encoder_83847:	�!
news_encoder_83849:	�&
news_encoder_83851:
��!
news_encoder_83853:	�
identity��$news_encoder/StatefulPartitionedCallI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
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
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:�����������
$news_encoder/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0news_encoder_83815news_encoder_83817news_encoder_83819news_encoder_83821news_encoder_83823news_encoder_83825news_encoder_83827news_encoder_83829news_encoder_83831news_encoder_83833news_encoder_83835news_encoder_83837news_encoder_83839news_encoder_83841news_encoder_83843news_encoder_83845news_encoder_83847news_encoder_83849news_encoder_83851news_encoder_83853* 
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*6
_read_only_resource_inputs
	
*<
config_proto,*

CPU

GPU 2J 8R(���������� *P
fKRI
G__inference_news_encoder_layer_call_and_return_conditional_losses_83391\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape-news_encoder/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������I
NoOpNoOp%^news_encoder/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�������������������: : : : : : : : : : : : : : : : : : : : 2L
$news_encoder/StatefulPartitionedCall$news_encoder/StatefulPartitionedCall:%!

_user_specified_name83853:%!

_user_specified_name83851:%!

_user_specified_name83849:%!

_user_specified_name83847:%!

_user_specified_name83845:%!

_user_specified_name83843:%!

_user_specified_name83841:%!

_user_specified_name83839:%!

_user_specified_name83837:%!

_user_specified_name83835:%
!

_user_specified_name83833:%	!

_user_specified_name83831:%!

_user_specified_name83829:%!

_user_specified_name83827:%!

_user_specified_name83825:%!

_user_specified_name83823:%!

_user_specified_name83821:%!

_user_specified_name83819:%!

_user_specified_name83817:%!

_user_specified_name83815:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_86302

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_dense_3_layer_call_and_return_conditional_losses_86603

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
@
input_15
serving_default_input_1:0���������#�
I
input_2>
serving_default_input_2:0�������������������G

activation9
StatefulPartitionedCall:0������������������tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	layer"
_tf_keras_layer
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_network
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
�
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
;14
<15
=16
>17
?18
@19
A20
B21
C22
D23
E24
F25"
trackable_list_wrapper
�
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
A14
B15
C16
D17
E18
F19"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ltrace_0
Mtrace_12�
%__inference_model_layer_call_fn_85076
%__inference_model_layer_call_fn_85134�
���
FullArgSpec)
args!�
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
annotations� *
 zLtrace_0zMtrace_1
�
Ntrace_0
Otrace_12�
@__inference_model_layer_call_and_return_conditional_losses_84876
@__inference_model_layer_call_and_return_conditional_losses_85018�
���
FullArgSpec)
args!�
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
annotations� *
 zNtrace_0zOtrace_1
�B�
 __inference__wrapped_model_82948input_1input_2"�
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
�
P
_variables
Q_iterations
R_learning_rate
S_index_dict
T
_momentums
U_velocities
V_update_step_xla"
experimentalOptimizer
,
Wserving_default"
signature_map
�
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
;14
<15
=16
>17
?18
@19"
trackable_list_wrapper
�
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
]trace_0
^trace_12�
2__inference_time_distributed_1_layer_call_fn_85251
2__inference_time_distributed_1_layer_call_fn_85296�
���
FullArgSpec)
args!�
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
annotations� *
 z]trace_0z^trace_1
�
_trace_0
`trace_12�
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_85453
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_85547�
���
FullArgSpec)
args!�
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
annotations� *
 z_trace_0z`trace_1
�
alayer-0
blayer_with_weights-0
blayer-1
clayer_with_weights-1
clayer-2
dlayer-3
elayer_with_weights-2
elayer-4
flayer_with_weights-3
flayer-5
glayer-6
hlayer_with_weights-4
hlayer-7
ilayer_with_weights-5
ilayer-8
jlayer-9
klayer_with_weights-6
klayer-10
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
	layer"
_tf_keras_layer
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses
AWQ
BWK
CWV"
_tf_keras_layer
�
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
DW
Eb
Fq"
_tf_keras_layer
�
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
;14
<15
=16
>17
?18
@19
A20
B21
C22
D23
E24
F25"
trackable_list_wrapper
�
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
A14
B15
C16
D17
E18
F19"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_user_encoder_layer_call_fn_84564
,__inference_user_encoder_layer_call_fn_84621�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_user_encoder_layer_call_and_return_conditional_losses_84325
G__inference_user_encoder_layer_call_and_return_conditional_losses_84507�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
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
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
#__inference_dot_layer_call_fn_85553�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
>__inference_dot_layer_call_and_return_conditional_losses_85563�
���
FullArgSpec
args�

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
annotations� *
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
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_activation_layer_call_fn_85568�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_activation_layer_call_and_return_conditional_losses_85573�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 :
��2dense/kernel
:�2
dense/bias
(:&�2batch_normalization/gamma
':%�2batch_normalization/beta
": 
��2dense_1/kernel
:�2dense_1/bias
*:(�2batch_normalization_1/gamma
):'�2batch_normalization_1/beta
": 
��2dense_2/kernel
:�2dense_2/bias
*:(�2batch_normalization_2/gamma
):'�2batch_normalization_2/beta
": 
��2dense_3/kernel
:�2dense_3/bias
0:.� (2batch_normalization/moving_mean
4:2� (2#batch_normalization/moving_variance
2:0� (2!batch_normalization_1/moving_mean
6:4� (2%batch_normalization_1/moving_variance
2:0� (2!batch_normalization_2/moving_mean
6:4� (2%batch_normalization_2/moving_variance
%:#
��2self_attention/WQ
%:#
��2self_attention/WK
%:#
��2self_attention/WV
 :
��2att_layer2/W
:�2att_layer2/b
:	�2att_layer2/q
J
;0
<1
=2
>3
?4
@5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_model_layer_call_fn_85076input_1input_2"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
%__inference_model_layer_call_fn_85134input_1input_2"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_84876input_1input_2"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_85018input_1input_2"�
���
FullArgSpec)
args!�
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
annotations� *
 
�
Q0
�1
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
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
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
�15
�16
�17
�18
�19"
trackable_list_wrapper
�
�0
�1
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
�15
�16
�17
�18
�19"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
#__inference_signature_wrapper_85206input_1input_2"�
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
J
;0
<1
=2
>3
?4
@5"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_time_distributed_1_layer_call_fn_85251inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
2__inference_time_distributed_1_layer_call_fn_85296inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_85453inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_85547inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
"
_tf_keras_input_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	/gamma
0beta
;moving_mean
<moving_variance"
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

1kernel
2bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	3gamma
4beta
=moving_mean
>moving_variance"
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

5kernel
6bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	7gamma
8beta
?moving_mean
@moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

9kernel
:bias"
_tf_keras_layer
�
-0
.1
/2
03
;4
<5
16
27
38
49
=10
>11
512
613
714
815
?16
@17
918
:19"
trackable_list_wrapper
�
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_news_encoder_layer_call_fn_83436
,__inference_news_encoder_layer_call_fn_83481�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_news_encoder_layer_call_and_return_conditional_losses_83322
G__inference_news_encoder_layer_call_and_return_conditional_losses_83391�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
�
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
;14
<15
=16
>17
?18
@19"
trackable_list_wrapper
�
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
0__inference_time_distributed_layer_call_fn_85618
0__inference_time_distributed_layer_call_fn_85663�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
K__inference_time_distributed_layer_call_and_return_conditional_losses_85820
K__inference_time_distributed_layer_call_and_return_conditional_losses_85914�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
5
A0
B1
C2"
trackable_list_wrapper
5
A0
B1
C2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_self_attention_layer_call_fn_85927�
���
FullArgSpec
args�
jQKVs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_self_attention_layer_call_and_return_conditional_losses_86056�
���
FullArgSpec
args�
jQKVs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
5
D0
E1
F2"
trackable_list_wrapper
5
D0
E1
F2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_att_layer2_layer_call_fn_86067
*__inference_att_layer2_layer_call_fn_86078�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkwjkwargs
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_att_layer2_layer_call_and_return_conditional_losses_86140
E__inference_att_layer2_layer_call_and_return_conditional_losses_86202�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkwjkwargs
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
J
;0
<1
=2
>3
?4
@5"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_user_encoder_layer_call_fn_84564input_5"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
,__inference_user_encoder_layer_call_fn_84621input_5"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
G__inference_user_encoder_layer_call_and_return_conditional_losses_84325input_5"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
G__inference_user_encoder_layer_call_and_return_conditional_losses_84507input_5"�
���
FullArgSpec)
args!�
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
annotations� *
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
#__inference_dot_layer_call_fn_85553inputs_0inputs_1"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
>__inference_dot_layer_call_and_return_conditional_losses_85563inputs_0inputs_1"�
���
FullArgSpec
args�

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
annotations� *
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
*__inference_activation_layer_call_fn_85568inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_activation_layer_call_and_return_conditional_losses_85573inputs"�
���
FullArgSpec
args�

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
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
�
�	variables
�	keras_api
�true_positives
�true_negatives
�false_positives
�false_negatives"
_tf_keras_metric
%:#
��2Adam/m/dense/kernel
%:#
��2Adam/v/dense/kernel
:�2Adam/m/dense/bias
:�2Adam/v/dense/bias
-:+�2 Adam/m/batch_normalization/gamma
-:+�2 Adam/v/batch_normalization/gamma
,:*�2Adam/m/batch_normalization/beta
,:*�2Adam/v/batch_normalization/beta
':%
��2Adam/m/dense_1/kernel
':%
��2Adam/v/dense_1/kernel
 :�2Adam/m/dense_1/bias
 :�2Adam/v/dense_1/bias
/:-�2"Adam/m/batch_normalization_1/gamma
/:-�2"Adam/v/batch_normalization_1/gamma
.:,�2!Adam/m/batch_normalization_1/beta
.:,�2!Adam/v/batch_normalization_1/beta
':%
��2Adam/m/dense_2/kernel
':%
��2Adam/v/dense_2/kernel
 :�2Adam/m/dense_2/bias
 :�2Adam/v/dense_2/bias
/:-�2"Adam/m/batch_normalization_2/gamma
/:-�2"Adam/v/batch_normalization_2/gamma
.:,�2!Adam/m/batch_normalization_2/beta
.:,�2!Adam/v/batch_normalization_2/beta
':%
��2Adam/m/dense_3/kernel
':%
��2Adam/v/dense_3/kernel
 :�2Adam/m/dense_3/bias
 :�2Adam/v/dense_3/bias
*:(
��2Adam/m/self_attention/WQ
*:(
��2Adam/v/self_attention/WQ
*:(
��2Adam/m/self_attention/WK
*:(
��2Adam/v/self_attention/WK
*:(
��2Adam/m/self_attention/WV
*:(
��2Adam/v/self_attention/WV
%:#
��2Adam/m/att_layer2/W
%:#
��2Adam/v/att_layer2/W
 :�2Adam/m/att_layer2/b
 :�2Adam/v/att_layer2/b
$:"	�2Adam/m/att_layer2/q
$:"	�2Adam/v/att_layer2/q
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
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
�
�trace_02�
%__inference_dense_layer_call_fn_86211�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
@__inference_dense_layer_call_and_return_conditional_losses_86222�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
<
/0
01
;2
<3"
trackable_list_wrapper
.
/0
01"
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
�
�trace_0
�trace_12�
3__inference_batch_normalization_layer_call_fn_86235
3__inference_batch_normalization_layer_call_fn_86248�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_86282
N__inference_batch_normalization_layer_call_and_return_conditional_losses_86302�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
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
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
'__inference_dropout_layer_call_fn_86307
'__inference_dropout_layer_call_fn_86312�
���
FullArgSpec!
args�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
B__inference_dropout_layer_call_and_return_conditional_losses_86324
B__inference_dropout_layer_call_and_return_conditional_losses_86329�
���
FullArgSpec!
args�
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
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
10
21"
trackable_list_wrapper
.
10
21"
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
�
�trace_02�
'__inference_dense_1_layer_call_fn_86338�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dense_1_layer_call_and_return_conditional_losses_86349�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
<
30
41
=2
>3"
trackable_list_wrapper
.
30
41"
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
�
�trace_0
�trace_12�
5__inference_batch_normalization_1_layer_call_fn_86362
5__inference_batch_normalization_1_layer_call_fn_86375�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_86409
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_86429�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
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
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
)__inference_dropout_1_layer_call_fn_86434
)__inference_dropout_1_layer_call_fn_86439�
���
FullArgSpec!
args�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
D__inference_dropout_1_layer_call_and_return_conditional_losses_86451
D__inference_dropout_1_layer_call_and_return_conditional_losses_86456�
���
FullArgSpec!
args�
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
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
50
61"
trackable_list_wrapper
.
50
61"
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
�
�trace_02�
'__inference_dense_2_layer_call_fn_86465�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dense_2_layer_call_and_return_conditional_losses_86476�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
<
70
81
?2
@3"
trackable_list_wrapper
.
70
81"
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
�
�trace_0
�trace_12�
5__inference_batch_normalization_2_layer_call_fn_86489
5__inference_batch_normalization_2_layer_call_fn_86502�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_86536
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_86556�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
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
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
)__inference_dropout_2_layer_call_fn_86561
)__inference_dropout_2_layer_call_fn_86566�
���
FullArgSpec!
args�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
D__inference_dropout_2_layer_call_and_return_conditional_losses_86578
D__inference_dropout_2_layer_call_and_return_conditional_losses_86583�
���
FullArgSpec!
args�
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
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
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
�
�trace_02�
'__inference_dense_3_layer_call_fn_86592�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dense_3_layer_call_and_return_conditional_losses_86603�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
J
;0
<1
=2
>3
?4
@5"
trackable_list_wrapper
n
a0
b1
c2
d3
e4
f5
g6
h7
i8
j9
k10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_news_encoder_layer_call_fn_83436input_4"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
,__inference_news_encoder_layer_call_fn_83481input_4"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
G__inference_news_encoder_layer_call_and_return_conditional_losses_83322input_4"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
G__inference_news_encoder_layer_call_and_return_conditional_losses_83391input_4"�
���
FullArgSpec)
args!�
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
annotations� *
 
J
;0
<1
=2
>3
?4
@5"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_time_distributed_layer_call_fn_85618inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
0__inference_time_distributed_layer_call_fn_85663inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
K__inference_time_distributed_layer_call_and_return_conditional_losses_85820inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
K__inference_time_distributed_layer_call_and_return_conditional_losses_85914inputs"�
���
FullArgSpec)
args!�
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
annotations� *
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
.__inference_self_attention_layer_call_fn_85927qkvs_0qkvs_1qkvs_2"�
���
FullArgSpec
args�
jQKVs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_self_attention_layer_call_and_return_conditional_losses_86056qkvs_0qkvs_1qkvs_2"�
���
FullArgSpec
args�
jQKVs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_att_layer2_layer_call_fn_86067inputs"�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkwjkwargs
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
*__inference_att_layer2_layer_call_fn_86078inputs"�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkwjkwargs
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
E__inference_att_layer2_layer_call_and_return_conditional_losses_86140inputs"�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkwjkwargs
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
E__inference_att_layer2_layer_call_and_return_conditional_losses_86202inputs"�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkwjkwargs
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
@
�0
�1
�2
�3"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:� (2true_positives
:� (2true_negatives
 :� (2false_positives
 :� (2false_negatives
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
%__inference_dense_layer_call_fn_86211inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
@__inference_dense_layer_call_and_return_conditional_losses_86222inputs"�
���
FullArgSpec
args�

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
annotations� *
 
.
;0
<1"
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
3__inference_batch_normalization_layer_call_fn_86235inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
3__inference_batch_normalization_layer_call_fn_86248inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_86282inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_86302inputs"�
���
FullArgSpec)
args!�
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
annotations� *
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
'__inference_dropout_layer_call_fn_86307inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
'__inference_dropout_layer_call_fn_86312inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_86324inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_86329inputs"�
���
FullArgSpec!
args�
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
annotations� *
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
'__inference_dense_1_layer_call_fn_86338inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
B__inference_dense_1_layer_call_and_return_conditional_losses_86349inputs"�
���
FullArgSpec
args�

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
annotations� *
 
.
=0
>1"
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
5__inference_batch_normalization_1_layer_call_fn_86362inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
5__inference_batch_normalization_1_layer_call_fn_86375inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_86409inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_86429inputs"�
���
FullArgSpec)
args!�
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
annotations� *
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
)__inference_dropout_1_layer_call_fn_86434inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
)__inference_dropout_1_layer_call_fn_86439inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
D__inference_dropout_1_layer_call_and_return_conditional_losses_86451inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
D__inference_dropout_1_layer_call_and_return_conditional_losses_86456inputs"�
���
FullArgSpec!
args�
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
annotations� *
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
'__inference_dense_2_layer_call_fn_86465inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
B__inference_dense_2_layer_call_and_return_conditional_losses_86476inputs"�
���
FullArgSpec
args�

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
annotations� *
 
.
?0
@1"
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
5__inference_batch_normalization_2_layer_call_fn_86489inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
5__inference_batch_normalization_2_layer_call_fn_86502inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_86536inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_86556inputs"�
���
FullArgSpec)
args!�
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
annotations� *
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
)__inference_dropout_2_layer_call_fn_86561inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
)__inference_dropout_2_layer_call_fn_86566inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
D__inference_dropout_2_layer_call_and_return_conditional_losses_86578inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
D__inference_dropout_2_layer_call_and_return_conditional_losses_86583inputs"�
���
FullArgSpec!
args�
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
annotations� *
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
'__inference_dense_3_layer_call_fn_86592inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
B__inference_dense_3_layer_call_and_return_conditional_losses_86603inputs"�
���
FullArgSpec
args�

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
annotations� *
 �
 __inference__wrapped_model_82948�-.</;012>3=456@7?89:ABCDEFk�h
a�^
\�Y
&�#
input_1���������#�
/�,
input_2�������������������
� "@�=
;

activation-�*

activation�������������������
E__inference_activation_layer_call_and_return_conditional_losses_85573q8�5
.�+
)�&
inputs������������������
� "5�2
+�(
tensor_0������������������
� �
*__inference_activation_layer_call_fn_85568f8�5
.�+
)�&
inputs������������������
� "*�'
unknown�������������������
E__inference_att_layer2_layer_call_and_return_conditional_losses_86140~DEFH�E
.�+
%�"
inputs���������#�

 
�

trainingp"-�*
#� 
tensor_0����������
� �
E__inference_att_layer2_layer_call_and_return_conditional_losses_86202~DEFH�E
.�+
%�"
inputs���������#�

 
�

trainingp "-�*
#� 
tensor_0����������
� �
*__inference_att_layer2_layer_call_fn_86067sDEFH�E
.�+
%�"
inputs���������#�

 
�

trainingp""�
unknown�����������
*__inference_att_layer2_layer_call_fn_86078sDEFH�E
.�+
%�"
inputs���������#�

 
�

trainingp ""�
unknown�����������
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_86409o=>348�5
.�+
!�
inputs����������
p

 
� "-�*
#� 
tensor_0����������
� �
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_86429o>3=48�5
.�+
!�
inputs����������
p 

 
� "-�*
#� 
tensor_0����������
� �
5__inference_batch_normalization_1_layer_call_fn_86362d=>348�5
.�+
!�
inputs����������
p

 
� ""�
unknown�����������
5__inference_batch_normalization_1_layer_call_fn_86375d>3=48�5
.�+
!�
inputs����������
p 

 
� ""�
unknown�����������
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_86536o?@788�5
.�+
!�
inputs����������
p

 
� "-�*
#� 
tensor_0����������
� �
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_86556o@7?88�5
.�+
!�
inputs����������
p 

 
� "-�*
#� 
tensor_0����������
� �
5__inference_batch_normalization_2_layer_call_fn_86489d?@788�5
.�+
!�
inputs����������
p

 
� ""�
unknown�����������
5__inference_batch_normalization_2_layer_call_fn_86502d@7?88�5
.�+
!�
inputs����������
p 

 
� ""�
unknown�����������
N__inference_batch_normalization_layer_call_and_return_conditional_losses_86282o;</08�5
.�+
!�
inputs����������
p

 
� "-�*
#� 
tensor_0����������
� �
N__inference_batch_normalization_layer_call_and_return_conditional_losses_86302o</;08�5
.�+
!�
inputs����������
p 

 
� "-�*
#� 
tensor_0����������
� �
3__inference_batch_normalization_layer_call_fn_86235d;</08�5
.�+
!�
inputs����������
p

 
� ""�
unknown�����������
3__inference_batch_normalization_layer_call_fn_86248d</;08�5
.�+
!�
inputs����������
p 

 
� ""�
unknown�����������
B__inference_dense_1_layer_call_and_return_conditional_losses_86349e120�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
'__inference_dense_1_layer_call_fn_86338Z120�-
&�#
!�
inputs����������
� ""�
unknown�����������
B__inference_dense_2_layer_call_and_return_conditional_losses_86476e560�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
'__inference_dense_2_layer_call_fn_86465Z560�-
&�#
!�
inputs����������
� ""�
unknown�����������
B__inference_dense_3_layer_call_and_return_conditional_losses_86603e9:0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
'__inference_dense_3_layer_call_fn_86592Z9:0�-
&�#
!�
inputs����������
� ""�
unknown�����������
@__inference_dense_layer_call_and_return_conditional_losses_86222e-.0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
%__inference_dense_layer_call_fn_86211Z-.0�-
&�#
!�
inputs����������
� ""�
unknown�����������
>__inference_dot_layer_call_and_return_conditional_losses_85563�i�f
_�\
Z�W
0�-
inputs_0�������������������
#� 
inputs_1����������
� "5�2
+�(
tensor_0������������������
� �
#__inference_dot_layer_call_fn_85553�i�f
_�\
Z�W
0�-
inputs_0�������������������
#� 
inputs_1����������
� "*�'
unknown�������������������
D__inference_dropout_1_layer_call_and_return_conditional_losses_86451e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
D__inference_dropout_1_layer_call_and_return_conditional_losses_86456e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
)__inference_dropout_1_layer_call_fn_86434Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
)__inference_dropout_1_layer_call_fn_86439Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
D__inference_dropout_2_layer_call_and_return_conditional_losses_86578e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
D__inference_dropout_2_layer_call_and_return_conditional_losses_86583e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
)__inference_dropout_2_layer_call_fn_86561Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
)__inference_dropout_2_layer_call_fn_86566Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
B__inference_dropout_layer_call_and_return_conditional_losses_86324e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
B__inference_dropout_layer_call_and_return_conditional_losses_86329e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
'__inference_dropout_layer_call_fn_86307Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
'__inference_dropout_layer_call_fn_86312Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
@__inference_model_layer_call_and_return_conditional_losses_84876�-.;</012=>3456?@789:ABCDEFs�p
i�f
\�Y
&�#
input_1���������#�
/�,
input_2�������������������
p

 
� "5�2
+�(
tensor_0������������������
� �
@__inference_model_layer_call_and_return_conditional_losses_85018�-.</;012>3=456@7?89:ABCDEFs�p
i�f
\�Y
&�#
input_1���������#�
/�,
input_2�������������������
p 

 
� "5�2
+�(
tensor_0������������������
� �
%__inference_model_layer_call_fn_85076�-.;</012=>3456?@789:ABCDEFs�p
i�f
\�Y
&�#
input_1���������#�
/�,
input_2�������������������
p

 
� "*�'
unknown�������������������
%__inference_model_layer_call_fn_85134�-.</;012>3=456@7?89:ABCDEFs�p
i�f
\�Y
&�#
input_1���������#�
/�,
input_2�������������������
p 

 
� "*�'
unknown�������������������
G__inference_news_encoder_layer_call_and_return_conditional_losses_83322�-.;</012=>3456?@789:9�6
/�,
"�
input_4����������
p

 
� "-�*
#� 
tensor_0����������
� �
G__inference_news_encoder_layer_call_and_return_conditional_losses_83391�-.</;012>3=456@7?89:9�6
/�,
"�
input_4����������
p 

 
� "-�*
#� 
tensor_0����������
� �
,__inference_news_encoder_layer_call_fn_83436u-.;</012=>3456?@789:9�6
/�,
"�
input_4����������
p

 
� ""�
unknown�����������
,__inference_news_encoder_layer_call_fn_83481u-.</;012>3=456@7?89:9�6
/�,
"�
input_4����������
p 

 
� ""�
unknown�����������
I__inference_self_attention_layer_call_and_return_conditional_losses_86056�ABC���
}�z
x�u
%�"
qkvs_0���������#�
%�"
qkvs_1���������#�
%�"
qkvs_2���������#�
� "1�.
'�$
tensor_0���������#�
� �
.__inference_self_attention_layer_call_fn_85927�ABC���
}�z
x�u
%�"
qkvs_0���������#�
%�"
qkvs_1���������#�
%�"
qkvs_2���������#�
� "&�#
unknown���������#��
#__inference_signature_wrapper_85206�-.</;012>3=456@7?89:ABCDEF|�y
� 
r�o
1
input_1&�#
input_1���������#�
:
input_2/�,
input_2�������������������"@�=
;

activation-�*

activation�������������������
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_85453�-.;</012=>3456?@789:E�B
;�8
.�+
inputs�������������������
p

 
� ":�7
0�-
tensor_0�������������������
� �
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_85547�-.</;012>3=456@7?89:E�B
;�8
.�+
inputs�������������������
p 

 
� ":�7
0�-
tensor_0�������������������
� �
2__inference_time_distributed_1_layer_call_fn_85251�-.;</012=>3456?@789:E�B
;�8
.�+
inputs�������������������
p

 
� "/�,
unknown��������������������
2__inference_time_distributed_1_layer_call_fn_85296�-.</;012>3=456@7?89:E�B
;�8
.�+
inputs�������������������
p 

 
� "/�,
unknown��������������������
K__inference_time_distributed_layer_call_and_return_conditional_losses_85820�-.;</012=>3456?@789:E�B
;�8
.�+
inputs�������������������
p

 
� ":�7
0�-
tensor_0�������������������
� �
K__inference_time_distributed_layer_call_and_return_conditional_losses_85914�-.</;012>3=456@7?89:E�B
;�8
.�+
inputs�������������������
p 

 
� ":�7
0�-
tensor_0�������������������
� �
0__inference_time_distributed_layer_call_fn_85618�-.;</012=>3456?@789:E�B
;�8
.�+
inputs�������������������
p

 
� "/�,
unknown��������������������
0__inference_time_distributed_layer_call_fn_85663�-.</;012>3=456@7?89:E�B
;�8
.�+
inputs�������������������
p 

 
� "/�,
unknown��������������������
G__inference_user_encoder_layer_call_and_return_conditional_losses_84325�-.;</012=>3456?@789:ABCDEF=�:
3�0
&�#
input_5���������#�
p

 
� "-�*
#� 
tensor_0����������
� �
G__inference_user_encoder_layer_call_and_return_conditional_losses_84507�-.</;012>3=456@7?89:ABCDEF=�:
3�0
&�#
input_5���������#�
p 

 
� "-�*
#� 
tensor_0����������
� �
,__inference_user_encoder_layer_call_fn_84564-.;</012=>3456?@789:ABCDEF=�:
3�0
&�#
input_5���������#�
p

 
� ""�
unknown�����������
,__inference_user_encoder_layer_call_fn_84621-.</;012>3=456@7?89:ABCDEF=�:
3�0
&�#
input_5���������#�
p 

 
� ""�
unknown����������