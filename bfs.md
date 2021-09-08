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

\1263. Minimum Moves to Move a Box to Their Target Location
Use visit to avoid duplication. Always move the person.

```python
class Solution:
    def minPushBox(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "B": box = (i, j)
                if grid[i][j] == "T": target = (i, j)
                if grid[i][j] == "S": person = (i, j)
        def distance(box):
            xTarget, yTarget = target
            xBox, yBox = box
            return abs(xTarget - xBox) + abs(yTarget - yBox)
        def isWall(position):
            x, y = position
            if x < 0 or x >= m: return True
            if y < 0 or y >= n: return True
            return grid[x][y] == "#"
        queue = [[distance(box), 0, person, box]]
        visited = {(person, box)}
        while queue:
            _, moves, person, box = heapq.heappop(queue)
            if (box == target):
                return moves
            for deltaX, deltaY in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                newPerson = (person[0] + deltaX, person[1] + deltaY)
                if isWall(newPerson): continue
                if(newPerson == box):
                    newBox = (box[0] + deltaX, box[1] + deltaY)
                    if (newPerson, newBox) in visited: continue
                    visited.add((newPerson, newBox))
                    if isWall(newBox): continue
                    heapq.heappush(queue, [distance(newBox) + moves + 1, moves + 1, newPerson, newBox])
                else:
                    if (newPerson, box) in visited: continue
                    visited.add((newPerson, box))
                    heapq.heappush(queue, [distance(box) + moves, moves, newPerson, box])
        return -1
```

