# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import get_type_hints, get_origin, get_args, Iterable, Any, Type, TypedDict
from collections.abc import Iterable as AbcIterable

from pydantic import BaseModel, create_model, Field, model_validator
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming

logger = logging.getLogger(__name__)


class _BaseModelWithIterableConversion(BaseModel):
    """Base model that auto-converts OpenAI's ValidatorIterator to lists.
    
    OpenAI's TypedDict uses Iterable types which create ValidatorIterator objects
    that don't serialize properly. This base class automatically converts them to
    lists after validation for proper JSON serialization.
    """
    
    @model_validator(mode='after')
    def _convert_iterables_to_lists(self) -> '_BaseModelWithIterableConversion':
        """Convert ValidatorIterator and other non-list iterables to lists.
        
        Returns:
            Self with all iterable fields converted to lists.
        """
        for field_name in self.__class__.model_fields.keys():
            value = getattr(self, field_name, None)
            # Convert any iterable (except strings, bytes, dicts, and already-lists) to list
            if value is not None and isinstance(value, AbcIterable) and not isinstance(value, (str, bytes, dict, list)):
                setattr(self, field_name, list(value))
        return self


def normalize_type(field_type: Any) -> Any:
    """Convert Iterable type annotations to list for better serialization.
    
    Args:
        field_type: The type annotation to normalize.
        
    Returns:
        list[T] if field_type is Iterable[T], otherwise returns field_type unchanged.
    """
    origin = get_origin(field_type)
    if origin is Iterable:
        args = get_args(field_type)
        return list[args[0]] if args else list[Any]
    return field_type


def create_pydantic_from_typeddict(typeddict_cls: Type[TypedDict]) -> Type[BaseModel]:
    """Dynamically create a Pydantic BaseModel from a TypedDict.
    
    Extracts field annotations from the TypedDict and creates a Pydantic model
    with automatic iterable-to-list conversion for proper JSON serialization.
    
    Args:
        typeddict_cls: The TypedDict class to convert (e.g., CompletionCreateParamsNonStreaming).
        
    Returns:
        A dynamically created Pydantic BaseModel with all fields from the TypedDict.
    """
    try:
        annotations = get_type_hints(typeddict_cls)
    except Exception:  # pylint: disable=broad-except
        # Fallback if get_type_hints fails on complex types
        logger.warning("Failed to get type hints for %s, falling back to __annotations__", typeddict_cls.__name__)
        annotations = getattr(typeddict_cls, '__annotations__', {})

    required_keys = getattr(typeddict_cls, '__required_keys__', set())

    fields = {}
    for field_name, field_type in annotations.items():
        # Normalize Iterable types to list
        normalized_type = normalize_type(field_type)
        
        if field_name in required_keys:
            # Required field
            fields[field_name] = (normalized_type, Field(...))
        else:
            # Optional field - use None as default
            fields[field_name] = (normalized_type, Field(default=None))

    # Create the model with auto-conversion of iterables
    model = create_model(
        'OpenAIChatRequest',
        __config__=None,
        __base__=_BaseModelWithIterableConversion,
        **fields,
        __module__=__name__
    )
    
    # Configure to allow extra fields from OpenAI spec
    model.model_config = {'extra': 'allow', 'arbitrary_types_allowed': True}
    return model


# Create the Pydantic model automatically from OpenAI's TypedDict
OpenAIChatRequest = create_pydantic_from_typeddict(CompletionCreateParamsNonStreaming)
