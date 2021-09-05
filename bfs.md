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

