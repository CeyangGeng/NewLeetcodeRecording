\1650. Lowest Common Ancestor of a Binary Tree III
The length of the two sub path is fixed.

```python
class Solution:
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        n1, n2 = p, q
        while n1 != n2:
            n1 = n1.parent if n1.parent else q
            n2 = n2.parent if n2.parent else p
        return n1
```

