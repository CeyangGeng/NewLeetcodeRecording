\426. Convert Binary Search Tree to Sorted Doubly Linked List

```python
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if not root: return None
        leftHead = self.treeToDoublyList(root.left)
        rightHead = self.treeToDoublyList(root.right)
        root.left = root
        root.right = root
        return self.connect((self.connect(leftHead, root)), rightHead)
    def connect(self, n1, n2):
        if not n1: return n2
        if not n2: return n1
        tail1, tail2 = n1.left, n2.left
        tail1.right = n2
        n2.left = tail1
        tail2.right = n1
        n1.left = tail2
        return n1
```

\3. Longest Substring Without Repeating Characters

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        res = 0
        left, right, size = 0, 0, len(s)
        dic = {}
        while right < size:
            char = s[right]
            dic[char] = dic.get(char, 0) + 1
            if dic[char] == 1: 
                res = max(res, right - left + 1)
            right += 1
            while dic[char] > 1:
                charLeft = s[left]
                dic[charLeft] -= 1
                left += 1
        return res
```

