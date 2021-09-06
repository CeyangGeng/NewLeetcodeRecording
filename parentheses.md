\301. Remove Invalid Parentheses
There are two key points to avoid duplication. The first one is to remove the first of a consecutive series of same ')'. The second ond is that the later removal must locate after the previous removal. To remove the left open parentheses, we need to reverse the string and the parentheses list. The last i records the next start of i, the last j records the next start of j. Since one character is removed, the i actually proceed by one in the new string.

```python
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        def helper(string, lis, last_i, last_j, par):
            count = 0
            for i in range(last_i, len(string)):
                char_i = string[i]
                if char_i == par[0]: count += 1
                if char_i == par[1]: count -= 1
                if count >= 0: continue
                for j in range(last_j, i + 1):
                    char_j = string[j]
                    if char_j == par[1] and (j == last_j or (j > last_j and string[j - 1] != par[1])):
                        helper(string[:j] + string[j + 1:], lis, i, j, par)
                return
            reverse = string[::-1]
            if par[0] == '(': 
                helper(reverse, lis, 0, 0, [')', '('])
            else: 
                lis.append(reverse)
        res = []
        helper(s, res, 0, 0, ['(', ')'])
        return res
```

