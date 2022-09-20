# Regex for Python
source: https://docs.python.org/3/library/re.html
```
In Unicode patterns (?a:...) switches to ASCII-only matching,
and (?u:...) switches to Unicode matching (default).
In byte pattern (?L:...) switches to locale depending matching,
and (?a:...) switches to ASCII-only matching (default).

\b
Matches the empty string, but only at the beginning or end of a word. A word is defined as a sequence of word characters. Note that formally, \b is defined as the boundary between a \w and a \W character (or vice versa), or between \w and the beginning/end of the string. This means that r'\bfoo\b' matches 'foo', 'foo.', '(foo)', 'bar foo baz' but not 'foobar' or 'foo3'.

\w
For Unicode (str) patterns:
Matches Unicode word characters; this includes most characters that can be part of a word in any language, as well as numbers and the underscore. If the ASCII flag is used, only [a-zA-Z0-9_] is matched.

For 8-bit (bytes) patterns:
Matches characters considered alphanumeric in the ASCII character set; this is equivalent to [a-zA-Z0-9_]. If the LOCALE flag is used, matches characters considered alphanumeric in the current locale and the underscore.

\W
Matches any character which is not a word character. This is the opposite of \w. If the ASCII flag is used this becomes the equivalent of [^a-zA-Z0-9_]. If the LOCALE flag is used, matches characters which are neither alphanumeric in the current locale nor the underscore.
```


