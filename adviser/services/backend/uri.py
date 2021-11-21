from typing import List
import re

_uri_pattern = re.compile(r"^([^\s\.#]+\.)*([^\s\.#]+)$") # URI components MUST NOT contain a ., # or whitespace characters and MUST NOT be empty
_uri_pattern_strict = re.compile(r"^([0-9a-z_]+\.)*([0-9a-z_]+)$") # URI components SHOULD only contain lower-case letters, digits and _


class URI:
	def __init__(self, uri: str) -> None:
		self._validate(uri)
		self.uri = uri 

	def components(self) -> List[str]:
		return self.uri.split('.')

	def _validate(self, uri: str):
		"""
		This function validates uri strings (e.g. topics) against the minmum AND strict rules for valid URI form, as specified below:

		* URIs are UTF-8 strings, following Java naming conventions (reversed url scheme), building a hierarchical namespace	
			* 	e.g. com.myapp.topic1
		* URI components are the (non-empty) string between dots
			* e.g. com, myapp and topic1 are compoenents
		* URI components MUST NOT contain a ., # or whitespace characters and MUST NOT be empty
		* URI components SHOULD only contain lower-case letters, digits and _ (enforced by default here)
		* URI components MUST NOT use wamp as a first URI component
		"""
		assert _uri_pattern.match(uri), f'URI {uri} does not comply to URI pattern (URI components MUST NOT contain a ., # or whitespace characters and MUST NOT be empty)'
		assert _uri_pattern_strict.match(uri), f'URI {uri} does not comply to STRICT URI pattern (URI components SHOULD only contain lower-case letters, digits and _)'
		assert not uri.startswith('wamp'), "URI MUST NOT use wamp as a first URI component"

