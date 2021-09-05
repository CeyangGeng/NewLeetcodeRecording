\953. Verifying an Alien Dictionary

Turn the order in to indices dictionary first.
Use zip(words, words[1:]) to pair the two adjacent elements
When comparing two words, there are two kinds of situations: 1) we get to know the order before finish iterating the shorter word 2) we can't tell the result after finish iterating the shorter word.  For the second situation, these two words have common head word, so if the first word is of longer length, we should return False. For the first situation, the result will come out while iterating over the part in the common length.

```python
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        indices = {}
        for i, c in enumerate(order):
            indices[c] = i
        for a, b in zip(words, words[1:]):
            if len(a) > len(b) and a[:len(b)] == b: return False
            for m, n in zip(a, b):
                if indices[m] < indices[n]: break
                elif indices[m] > indices[n]: return False
        return True
```

\1570. Dot Product of Two Sparse Vectors
Use the index to record the non zero element and position.

```python
class SparseVector:
    def __init__(self, nums: List[int]):
        self.indices = {}
        for i, num in enumerate(nums):
            if num != 0:
                self.indices[i] = num
        

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        res = 0
        for i, n in vec.indices.items():
            if i in self.indices.keys():
                res += n * self.indices[i]
        return res
```

\1428. Leftmost Column with at Least a One
Iterate start from the upper right corner. If the current element is 1, we can continue to decrease the column index unitil we find 0 in this row. After finding 0 in a row, we can increase the row index to see if this column in the next row is 1. If the element in the current column and next row is 0, it indicates that the first appearance of 1 in this next row is behind of the current column. If the element in the current column and next row is 1, it indicates that the first appearance of 1 in the next row is before the current column.

```python
class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        row, col = binaryMatrix.dimensions()
        i, j, res = 0, col - 1, -1
        while i < row and j >= 0:
            if binaryMatrix.get(i, j) == 0:
                i += 1
            elif binaryMatrix.get(i, j) == 1:
                res = j
                j -= 1
        return res
```

