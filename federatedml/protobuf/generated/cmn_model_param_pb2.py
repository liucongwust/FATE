# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: cmn-model-param.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='cmn-model-param.proto',
  package='com.webank.ai.fate.core.mlmodel.buffer.cmn',
  syntax='proto3',
  serialized_options=_b('B\022CMNModelParamProto'),
  serialized_pb=_b('\n\x15\x63mn-model-param.proto\x12*com.webank.ai.fate.core.mlmodel.buffer.cmn\"\x15\n\x06IDList\x12\x0b\n\x03ids\x18\x01 \x03(\x05\"\xaa\x04\n\rCMNModelParam\x12\x16\n\x0e\x61ggregate_iter\x18\x01 \x01(\x05\x12\x19\n\x11saved_model_bytes\x18\x02 \x01(\x0c\x12\x14\n\x0closs_history\x18\x03 \x03(\x01\x12\x14\n\x0cis_converged\x18\x04 \x01(\x08\x12\x0e\n\x06header\x18\x05 \x03(\t\x12\x10\n\x08user_num\x18\x06 \x01(\x05\x12\x10\n\x08item_num\x18\x07 \x01(\x05\x12\\\n\nuser_items\x18\x08 \x03(\x0b\x32H.com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.UserItemsEntry\x12\\\n\nitem_users\x18\t \x03(\x0b\x32H.com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.ItemUsersEntry\x1a\x64\n\x0eUserItemsEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\x41\n\x05value\x18\x02 \x01(\x0b\x32\x32.com.webank.ai.fate.core.mlmodel.buffer.cmn.IDList:\x02\x38\x01\x1a\x64\n\x0eItemUsersEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\x41\n\x05value\x18\x02 \x01(\x0b\x32\x32.com.webank.ai.fate.core.mlmodel.buffer.cmn.IDList:\x02\x38\x01\x42\x14\x42\x12\x43MNModelParamProtob\x06proto3')
)




_IDLIST = _descriptor.Descriptor(
  name='IDList',
  full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.IDList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ids', full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.IDList.ids', index=0,
      number=1, type=5, cpp_type=1, label=3,
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
  serialized_start=69,
  serialized_end=90,
)


_CMNMODELPARAM_USERITEMSENTRY = _descriptor.Descriptor(
  name='UserItemsEntry',
  full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.UserItemsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.UserItemsEntry.key', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.UserItemsEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=445,
  serialized_end=545,
)

_CMNMODELPARAM_ITEMUSERSENTRY = _descriptor.Descriptor(
  name='ItemUsersEntry',
  full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.ItemUsersEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.ItemUsersEntry.key', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.ItemUsersEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=547,
  serialized_end=647,
)

_CMNMODELPARAM = _descriptor.Descriptor(
  name='CMNModelParam',
  full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='aggregate_iter', full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.aggregate_iter', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='saved_model_bytes', full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.saved_model_bytes', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='loss_history', full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.loss_history', index=2,
      number=3, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='is_converged', full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.is_converged', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='header', full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.header', index=4,
      number=5, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='user_num', full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.user_num', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='item_num', full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.item_num', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='user_items', full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.user_items', index=7,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='item_users', full_name='com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.item_users', index=8,
      number=9, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_CMNMODELPARAM_USERITEMSENTRY, _CMNMODELPARAM_ITEMUSERSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=93,
  serialized_end=647,
)

_CMNMODELPARAM_USERITEMSENTRY.fields_by_name['value'].message_type = _IDLIST
_CMNMODELPARAM_USERITEMSENTRY.containing_type = _CMNMODELPARAM
_CMNMODELPARAM_ITEMUSERSENTRY.fields_by_name['value'].message_type = _IDLIST
_CMNMODELPARAM_ITEMUSERSENTRY.containing_type = _CMNMODELPARAM
_CMNMODELPARAM.fields_by_name['user_items'].message_type = _CMNMODELPARAM_USERITEMSENTRY
_CMNMODELPARAM.fields_by_name['item_users'].message_type = _CMNMODELPARAM_ITEMUSERSENTRY
DESCRIPTOR.message_types_by_name['IDList'] = _IDLIST
DESCRIPTOR.message_types_by_name['CMNModelParam'] = _CMNMODELPARAM
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

IDList = _reflection.GeneratedProtocolMessageType('IDList', (_message.Message,), dict(
  DESCRIPTOR = _IDLIST,
  __module__ = 'cmn_model_param_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.cmn.IDList)
  ))
_sym_db.RegisterMessage(IDList)

CMNModelParam = _reflection.GeneratedProtocolMessageType('CMNModelParam', (_message.Message,), dict(

  UserItemsEntry = _reflection.GeneratedProtocolMessageType('UserItemsEntry', (_message.Message,), dict(
    DESCRIPTOR = _CMNMODELPARAM_USERITEMSENTRY,
    __module__ = 'cmn_model_param_pb2'
    # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.UserItemsEntry)
    ))
  ,

  ItemUsersEntry = _reflection.GeneratedProtocolMessageType('ItemUsersEntry', (_message.Message,), dict(
    DESCRIPTOR = _CMNMODELPARAM_ITEMUSERSENTRY,
    __module__ = 'cmn_model_param_pb2'
    # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam.ItemUsersEntry)
    ))
  ,
  DESCRIPTOR = _CMNMODELPARAM,
  __module__ = 'cmn_model_param_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.cmn.CMNModelParam)
  ))
_sym_db.RegisterMessage(CMNModelParam)
_sym_db.RegisterMessage(CMNModelParam.UserItemsEntry)
_sym_db.RegisterMessage(CMNModelParam.ItemUsersEntry)


DESCRIPTOR._options = None
_CMNMODELPARAM_USERITEMSENTRY._options = None
_CMNMODELPARAM_ITEMUSERSENTRY._options = None
# @@protoc_insertion_point(module_scope)