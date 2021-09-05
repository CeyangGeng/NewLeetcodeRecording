\1249. Minimum Remove to Make Valid Parentheses

- Using stack to resolve this problem. While meeting with the close parenthesis, pop out the open parenthesis. If the stack is empty, indicating that this close parenthesis has no matching open parenthesis, so we need to make this close parenthesis empty. If there still are some opening parenthesis after iterating over the whole string, the left open parenthesis are wrong signs, we also need to make this character empty.

```python
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        s = list(s)
        stack = []
        for i, char in enumerate(s):
            if char == '(': stack.append(i)
            elif char == ')': 
                if stack: stack.pop()
                else: s[i] = ''
        while stack: 
            s[stack.pop()] = ''
        return ''.join(s)
```

\1762. Buildings With an Ocean View
The iterating start from the right most building. The right most building is sure to have the ocean view. The buildings have an ocean view are in decrease order.

```python
class Solution:
    def findBuildings(self, heights: List[int]) -> List[int]:
        size = len(heights)
        stack = [size - 1]
        for i in range(size - 2, -1, -1):
            if heights[i] > heights[stack[-1]]:
                stack.append(i)
        return stack[::-1]
```

