# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ncf-model-param.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='ncf-model-param.proto',
  package='com.webank.ai.fate.core.mlmodel.buffer.ncf',
  syntax='proto3',
  serialized_options=_b('B\022NCFModelParamProto'),
  serialized_pb=_b('\n\x15ncf-model-param.proto\x12*com.webank.ai.fate.core.mlmodel.buffer.ncf\"\xa2\x01\n\rNCFModelParam\x12\x16\n\x0e\x61ggregate_iter\x18\x01 \x01(\x05\x12\x19\n\x11saved_model_bytes\x18\x02 \x01(\x0c\x12\x14\n\x0closs_history\x18\x03 \x03(\x01\x12\x14\n\x0cis_converged\x18\x04 \x01(\x08\x12\x0e\n\x06header\x18\x05 \x03(\t\x12\x10\n\x08user_ids\x18\x06 \x03(\x05\x12\x10\n\x08item_ids\x18\x07 \x03(\x05\x42\x14\x42\x12NCFModelParamProtob\x06proto3')
)




_NCFMODELPARAM = _descriptor.Descriptor(
  name='NCFModelParam',
  full_name='com.webank.ai.fate.core.mlmodel.buffer.ncf.NCFModelParam',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='aggregate_iter', full_name='com.webank.ai.fate.core.mlmodel.buffer.ncf.NCFModelParam.aggregate_iter', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='saved_model_bytes', full_name='com.webank.ai.fate.core.mlmodel.buffer.ncf.NCFModelParam.saved_model_bytes', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='loss_history', full_name='com.webank.ai.fate.core.mlmodel.buffer.ncf.NCFModelParam.loss_history', index=2,
      number=3, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='is_converged', full_name='com.webank.ai.fate.core.mlmodel.buffer.ncf.NCFModelParam.is_converged', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='header', full_name='com.webank.ai.fate.core.mlmodel.buffer.ncf.NCFModelParam.header', index=4,
      number=5, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='user_ids', full_name='com.webank.ai.fate.core.mlmodel.buffer.ncf.NCFModelParam.user_ids', index=5,
      number=6, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='item_ids', full_name='com.webank.ai.fate.core.mlmodel.buffer.ncf.NCFModelParam.item_ids', index=6,
      number=7, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=70,
  serialized_end=232,
)

DESCRIPTOR.message_types_by_name['NCFModelParam'] = _NCFMODELPARAM
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

NCFModelParam = _reflection.GeneratedProtocolMessageType('NCFModelParam', (_message.Message,), dict(
  DESCRIPTOR = _NCFMODELPARAM,
  __module__ = 'ncf_model_param_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.ncf.NCFModelParam)
  ))
_sym_db.RegisterMessage(NCFModelParam)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
