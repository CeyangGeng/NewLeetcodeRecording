\680. Valid Palindrome II
Move left and right pointers inwards. If the characters at the left and right pointers are not the same, we can either remove the character at the left or the right. Then check if the left part is palindrome string.

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        left, right = 0, len(s) - 1
        while left < right:
            if s[left] != s[right]:
                a, b = s[left: right], s[left + 1: right + 1]
                return a == a[::-1] or b == b[::-1]
            left, right = left + 1, right - 1
        return True
```

