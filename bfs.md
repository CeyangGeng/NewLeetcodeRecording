\994. Rotting Oranges
Use bfs to rotten the oranges, the element put into the queue should include the coordination and the time. After bfs, if there are still fresh oranges, return -1, otherwise return rotten time.

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        res = 0
        queue = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    queue.append((i, j, 0))
        while queue:
            x, y, time = queue.pop(0)
            res = max(res, time)
            for deltaX, deltaY in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                newX, newY = x + deltaX, y + deltaY
                if 0 <= newX < m and 0 <= newY < n and grid[newX][newY] == 1:
                    grid[newX][newY] = 2
                    queue.append((newX, newY, time + 1))
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1: return -1
        return res
```

\199. Binary Tree Right Side View
In each while loop, create a tmp which include the new level nodes, at the end of the while loop, replace the queue with this tmp list.

```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root: return []
        res = []
        queue = [root]
        while queue:
            res.append(queue[-1].val)
            tmp = []
            for element in queue:
                if element.left: tmp.append(element.left)
                if element.right: tmp.append(element.right)
            queue = tmp
        return res
```

\238. Product of Array Except Self
From left to right, use the list to record the pre product. From right to left, use a variable to record the post product.

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1]
        for i in range(1, len(nums)):
            res.append(res[-1] * nums[i - 1])
        right = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= right
            right *= nums[i]
        return res
```

