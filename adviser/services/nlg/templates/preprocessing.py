###############################################################################
#
# Copyright 2020, University of Stuttgart: Institute for Natural Language Processing (IMS)
#
# This file is part of Adviser.
# Adviser is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3.
#
# Adviser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Adviser.  If not, see <https://www.gnu.org/licenses/>.
#
###############################################################################

from typing import List


def _detect_colon(line: str) -> int:
    in_string = False
    in_string_escape = False
    for i, character in enumerate(line):
        character = line[i]
        if in_string_escape:
            in_string_escape = False
        elif in_string:
            if character == '"':
                in_string = False
            elif character == '\\':
                in_string_escape = True
        else:
            if character == '"':
                in_string = True
            elif character == ':':
                return i
    return -1


def _get_block_level(line: str) -> int:
    whitespace_count = 0
    for character in line:
        if character != ' ':
            break
        whitespace_count += 1
    if whitespace_count % 4 != 0:
        raise BaseException('Block indentation failed! Block indent must contain 4 spaces!')
    return whitespace_count // 4


class _Preprocessor:
    def __init__(self, filename):
        self._lines: List[str] = []
        self._content = ''
        with open(filename, 'r', encoding='utf8') as file:
            self._content = file.read()
    
    def _preprocess(self):
        self._replace_tabs_with_spaces()
        self._lines = self._content.split('\n')
        self._remove_comments()
        self._replace_colons_with_block_indents()
        self._remove_empty_lines()
        self._add_message_keyword_in_front_of_strings()

    def get_preprocessed_lines(self):
        self._preprocess()
        return self._lines

    def _replace_tabs_with_spaces(self):
        self._content = self._content.replace('\t', '    ')
    
    def _remove_comments(self):
        lines_with_content: List[str] = []
        for line in self._lines:
            if not line.strip().startswith('#'):
                lines_with_content.append(line)
        self._lines = lines_with_content

    def _remove_empty_lines(self):
        lines_with_content: List[str] = []
        for line in self._lines:
            if line.strip() != '':
                lines_with_content.append(line)
        self._lines = lines_with_content

    def _add_message_keyword_in_front_of_strings(self):
        for i, line in enumerate(self._lines):
            if line.strip().startswith('"'):
                block_level = _get_block_level(line)
                self._lines[i] = ' ' * (4 * block_level) + 'message ' + line[4 * block_level:]

    def _replace_colons_with_block_indents(self):
        updated_lines: List[str] = []
        for line in self._lines:
            colon_index = _detect_colon(line)
            if colon_index >= 0:
                block_level = _get_block_level(line)
                updated_lines.append(line[:colon_index])
                updated_lines.append(' ' * ((block_level + 1) * 4) + line[colon_index+1:].strip())
            else:
                updated_lines.append(line)
        self._lines = updated_lines
