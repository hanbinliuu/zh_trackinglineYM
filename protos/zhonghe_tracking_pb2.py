# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: zhonghe_tracking.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='zhonghe_tracking.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x16zhonghe_tracking.proto\"\xa9\x02\n\x0fTrackingRequest\x12\x14\n\x0crequest_flag\x18\x01 \x01(\x05\x12\x11\n\ttimestamp\x18\x02 \x01(\x01\x12\r\n\x05image\x18\x03 \x01(\x0c\x12\x19\n\x11vision_move_speed\x18\x04 \x01(\x02\x12\x1f\n\x17vision_move_to_location\x18\x05 \x01(\x02\x12\x10\n\x08global_x\x18\x06 \x01(\x02\x12\x10\n\x08global_y\x18\x07 \x01(\x02\x12\x17\n\x0fglobal_distance\x18\x08 \x01(\x02\x12\x15\n\rpart_distance\x18\t \x01(\x02\x12\x14\n\x0cwidth_target\x18\n \x01(\x02\x12\x11\n\tangle_yaw\x18\x0b \x01(\x02\x12\x11\n\twidth_yaw\x18\x0c \x01(\x02\x12\x12\n\nreset_flag\x18\r \x01(\x05\"\xbe\x03\n\x10TrackingResponse\x12\x15\n\rresponse_flag\x18\x01 \x01(\x05\x12\x11\n\ttimestamp\x18\x02 \x01(\x02\x12\x1a\n\x12\x63urrent_move_speed\x18\x03 \x01(\x02\x12\x18\n\x10\x63urrent_location\x18\x04 \x01(\x02\x12\x15\n\rtarget_offset\x18\x05 \x01(\x02\x12\x10\n\x08global_x\x18\x06 \x01(\x02\x12\x10\n\x08global_y\x18\x07 \x01(\x02\x12\x17\n\x0fglobal_distance\x18\x08 \x01(\x02\x12\x15\n\rpart_distance\x18\t \x01(\x02\x12\x13\n\x0bgyroscope_x\x18\n \x03(\x02\x12\x13\n\x0bgyroscope_y\x18\x0b \x03(\x02\x12\x13\n\x0bgyroscope_z\x18\x0c \x03(\x02\x12\x0c\n\x04roll\x18\r \x03(\x02\x12\r\n\x05pitch\x18\x0e \x03(\x02\x12\x0f\n\x07heading\x18\x0f \x03(\x02\x12\x14\n\x0cwidth_target\x18\x10 \x01(\x02\x12\x10\n\x08odometer\x18\x11 \x03(\x02\x12\x15\n\rlast_odometer\x18\x12 \x01(\x02\x12\x12\n\nreset_flag\x18\x13 \x01(\x05\x12\x1f\n\x17vision_move_to_location\x18\x14 \x01(\x02\x32K\n\x0fZhongheTracking\x12\x38\n\x11TrackingOperation\x12\x10.TrackingRequest\x1a\x11.TrackingResponseb\x06proto3'
)




_TRACKINGREQUEST = _descriptor.Descriptor(
  name='TrackingRequest',
  full_name='TrackingRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='request_flag', full_name='TrackingRequest.request_flag', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='TrackingRequest.timestamp', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='image', full_name='TrackingRequest.image', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vision_move_speed', full_name='TrackingRequest.vision_move_speed', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vision_move_to_location', full_name='TrackingRequest.vision_move_to_location', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='global_x', full_name='TrackingRequest.global_x', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='global_y', full_name='TrackingRequest.global_y', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='global_distance', full_name='TrackingRequest.global_distance', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='part_distance', full_name='TrackingRequest.part_distance', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='width_target', full_name='TrackingRequest.width_target', index=9,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='angle_yaw', full_name='TrackingRequest.angle_yaw', index=10,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='width_yaw', full_name='TrackingRequest.width_yaw', index=11,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='reset_flag', full_name='TrackingRequest.reset_flag', index=12,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=27,
  serialized_end=324,
)


_TRACKINGRESPONSE = _descriptor.Descriptor(
  name='TrackingResponse',
  full_name='TrackingResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='response_flag', full_name='TrackingResponse.response_flag', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='TrackingResponse.timestamp', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='current_move_speed', full_name='TrackingResponse.current_move_speed', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='current_location', full_name='TrackingResponse.current_location', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='target_offset', full_name='TrackingResponse.target_offset', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='global_x', full_name='TrackingResponse.global_x', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='global_y', full_name='TrackingResponse.global_y', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='global_distance', full_name='TrackingResponse.global_distance', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='part_distance', full_name='TrackingResponse.part_distance', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='gyroscope_x', full_name='TrackingResponse.gyroscope_x', index=9,
      number=10, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='gyroscope_y', full_name='TrackingResponse.gyroscope_y', index=10,
      number=11, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='gyroscope_z', full_name='TrackingResponse.gyroscope_z', index=11,
      number=12, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='roll', full_name='TrackingResponse.roll', index=12,
      number=13, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='pitch', full_name='TrackingResponse.pitch', index=13,
      number=14, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='heading', full_name='TrackingResponse.heading', index=14,
      number=15, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='width_target', full_name='TrackingResponse.width_target', index=15,
      number=16, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='odometer', full_name='TrackingResponse.odometer', index=16,
      number=17, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='last_odometer', full_name='TrackingResponse.last_odometer', index=17,
      number=18, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='reset_flag', full_name='TrackingResponse.reset_flag', index=18,
      number=19, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vision_move_to_location', full_name='TrackingResponse.vision_move_to_location', index=19,
      number=20, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=327,
  serialized_end=773,
)

DESCRIPTOR.message_types_by_name['TrackingRequest'] = _TRACKINGREQUEST
DESCRIPTOR.message_types_by_name['TrackingResponse'] = _TRACKINGRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TrackingRequest = _reflection.GeneratedProtocolMessageType('TrackingRequest', (_message.Message,), {
  'DESCRIPTOR' : _TRACKINGREQUEST,
  '__module__' : 'zhonghe_tracking_pb2'
  # @@protoc_insertion_point(class_scope:TrackingRequest)
  })
_sym_db.RegisterMessage(TrackingRequest)

TrackingResponse = _reflection.GeneratedProtocolMessageType('TrackingResponse', (_message.Message,), {
  'DESCRIPTOR' : _TRACKINGRESPONSE,
  '__module__' : 'zhonghe_tracking_pb2'
  # @@protoc_insertion_point(class_scope:TrackingResponse)
  })
_sym_db.RegisterMessage(TrackingResponse)



_ZHONGHETRACKING = _descriptor.ServiceDescriptor(
  name='ZhongheTracking',
  full_name='ZhongheTracking',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=775,
  serialized_end=850,
  methods=[
  _descriptor.MethodDescriptor(
    name='TrackingOperation',
    full_name='ZhongheTracking.TrackingOperation',
    index=0,
    containing_service=None,
    input_type=_TRACKINGREQUEST,
    output_type=_TRACKINGRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_ZHONGHETRACKING)

DESCRIPTOR.services_by_name['ZhongheTracking'] = _ZHONGHETRACKING

# @@protoc_insertion_point(module_scope)