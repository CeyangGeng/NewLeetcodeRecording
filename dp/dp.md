> ## The general solution for this is think about how to apply the question to the base case, how to get the transfer function.

> 
>
>  Minimum Path Sum  https://leetcode.com/problems/minimum-path-sum/
>
> > Solution
> >
> > > Bottom up
> > >
> > > ```python
> > > class Solution:
> > >     def minPathSum(self, grid: List[List[int]]) -> int:
> > >         m, n = len(grid), len(grid[0])
> > >         for i in range(1, m):
> > >             grid[i][0] += grid[i - 1][0]
> > >         for j in range(1, n):
> > >             grid[0][j] += grid[0][j - 1]
> > >         for i in range(1, m):
> > >             for j in range(1, n):
> > >                 grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
> > >         return grid[m - 1][n - 1]
> > > ```
> > >
> > > Top down
> > >
> > > ```python
> > > class Solution:
> > >     def minPathSum(self, grid: List[List[int]]) -> int:
> > >         memo = dict()
> > >         m, n = len(grid), len(grid[0])
> > >         def helper(i, j, dic):
> > >             if i == 0 and j == 0: return grid[0][0]
> > >             if i == 0 : return grid[i][j] + helper(i, j - 1, dic)
> > >             if j == 0 : return grid[i][j] + helper(i - 1, j, dic)
> > >             if (i, j) in dic: return dic[(i, j)]
> > >             result = grid[i][j] + min(helper(i - 1, j, dic), helper(i, j - 1, dic))
> > >             dic[(i, j)] = result
> > >             return result
> > >         return helper(m - 1, n - 1, memo)
> > > ```
>
> Triangle https://leetcode.com/problems/triangle/
>
> > Solution
> >
> > > Initial Bottom up
> > >
> > > ```python
> > > class Solution:
> > >     def minimumTotal(self, triangle: List[List[int]]) -> int:
> > >         size = len(triangle)
> > >         dp = [triangle[0][0]] + [0 for _ in range(size - 1)]
> > >         for i in range(1, size):
> > >             for j in range(i + 1):
> > >                 minAdjacent = float('inf')
> > >                 if j - 1 > -1: minAdjacent = min(minAdjacent, triangle[i - 1][j - 1])
> > >                 if j < i: minAdjacent = min(minAdjacent, triangle[i - 1][j])
> > >                 triangle[i][j] += minAdjacent
> > >         return min(triangle[-1])
> > > ```
> > >
> > > Using O(n) extra space
> > >
> > > ```python
> > > class Solution:
> > >     def minimumTotal(self, triangle: List[List[int]]) -> int:
> > >         size = len(triangle)
> > >         dp = [triangle[0][0]] + [0 for _ in range(size - 1)]
> > >         for i in range(1, size):
> > >             for j in range(i, -1, -1):
> > >                 minAdjacent = float('inf')
> > >                 if j - 1 > -1: minAdjacent = min(minAdjacent, dp[j - 1])
> > >                 if j < i: minAdjacent = min(minAdjacent, dp[j])
> > >                 dp[j] = minAdjacent + triangle[i][j]
> > >         print(dp)
> > >         return min(dp)
> > > ```
> > >
> > >  Bottom up
> > >
> > > ```python
> > > class Solution:
> > >     def minimumTotal(self, triangle: List[List[int]]) -> int:
> > >         dp = triangle[-1].copy()
> > >         print(dp)
> > >         size = len(triangle)
> > >         for i in range(size -2, -1, -1):
> > >             for j in range(i + 1):
> > >                 dp[j] = min(dp[j], dp[j + 1]) + triangle[i][j]
> > >         return dp[0]
> > > ```
>
> Dungeon Game https://leetcode.com/problems/dungeon-game/
>
> Solution
>
> > 1. Top down 
> >
> >    > Top-down template
> >    >
> >    > Def overallFunc()
> >    >
> >    > ​	getheight, getwidth
> >    >
> >    > ​	initialize memo
> >    >
> >    > ​	helperFunc(i, j, memo)
> >    >
> >    > ​		baseCase
> >    >
> >    > ​		res = transferFunction
> >    >
> >    > ​		storeResIntoMemo
> >    >
> >    > ​		return res
> >    >
> >    > ​	helper(0, 0, memo)
> >    >
> >    > > Solution
> >    > >
> >    > > ```python
> >    > > class Solution:
> >    > >     def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
> >    > >         
> >    > >         m, n = len(dungeon), len(dungeon[0])
> >    > >         memo = [[float('inf') for _ in range(n)] for _ in range(m)]
> >    > >         
> >    > >         def helper(i, j, memo):
> >    > >             if i == m or j == n: return float('inf')
> >    > >             if i == m - 1 and j == n - 1 : return 1 if dungeon[i][j] > 0 else 1 - dungeon[i][j]
> >    > >             if memo[i][j] < float('inf'): return memo[i][j]
> >    > >             goRight = helper(i, j + 1, memo)
> >    > >             goDown = helper(i + 1, j, memo)
> >    > >             tmp = min(goRight, goDown) - dungeon[i][j]
> >    > >             res = tmp if tmp > 0 else 1
> >    > >             memo[i][j] = res
> >    > >             return res
> >    > >             
> >    > >         return helper(0, 0, memo)
> >    > > ```
> >
> > 2. Bottom up
> >
> >    ```python
> >    class Solution:
> >        def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
> >            
> >            m, n = len(dungeon), len(dungeon[0])
> >            dp = [[float('inf') for _ in range(n + 1)] for _ in range(m + 1)]
> >            dp[m][n - 1], dp[m - 1][n] = 1, 1
> >            for i in range(m - 1, -1, -1):
> >                for j in range(n - 1, -1, -1):
> >                    tmp = min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j]
> >                    dp[i][j] = tmp if tmp > 0 else 1
> >            return dp[0][0]
> >    ```
>
> Maximal Square https://leetcode.com/problems/maximal-square/
>
> > 1. Initial bottom up solution
> >
> >    ```python
> >    class Solution:
> >        def maximalSquare(self, matrix: List[List[str]]) -> int:
> >            maxLen = 0
> >            m, n = len(matrix), len(matrix[0])
> >            dp = [[0 for _ in range(n)] for _ in range(m)]
> >            for i in range(m):
> >                if matrix[i][0] == "1":
> >                    dp[i][0] = 1
> >                    maxLen = 1
> >            for j in range(n):
> >                if matrix[0][j] == "1":
> >                    dp[0][j] = 1
> >                    maxLen = 1
> >            for i in range(1, m):
> >                for j in range(1, n):
> >                    if matrix[i][j] == "1":
> >                        dp[i][j] = 1 + min(dp[i - 1][j], dp[i - 1][j - 1], dp[i][j - 1])
> >                        maxLen = max(maxLen, dp[i][j])
> >            return maxLen ** 2
> >    ```
> >
> > 2. Extend the edge of the dp array
> >
> >    ```python
> >    class Solution:
> >        def maximalSquare(self, matrix: List[List[str]]) -> int:
> >            maxLen = 0
> >            m, n = len(matrix), len(matrix[0])
> >            dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
> >            for i in range(1, m + 1):
> >                for j in range(1, n + 1):
> >                    if matrix[i - 1][j - 1] == "1":
> >                        dp[i][j] = 1 + min(dp[i - 1][j], dp[i - 1][j - 1], dp[i][j - 1])
> >                        maxLen = max(maxLen, dp[i][j])
> >            return maxLen ** 2
> >    ```
>
> Perfect Squares https://leetcode.com/problems/perfect-squares/
>
> > BFS
> >
> > If you find that during the recursive, the partial count might be larger than the whole count, you should consider using the bfs which could find the first terminal quickly.
> >
> > ```PYTHON
> > class Solution:
> >     def numSquares(self, n: int) -> int:
> >         if n <= 1: return n
> >         maxFactor = int(sqrt(n))
> >         squares = []
> >         i = 1
> >         while i **2 <= n:
> >             squares.append(i ** 2)
> >             i += 1
> >         queue = {n}
> >         count = 0
> >         while queue:
> >             count += 1
> >             temp = set()
> >             for number in queue:
> >                 for square in squares:
> >                     if square == number: return count
> >                     if square > number: break
> >                     temp.add(number - square)
> >             queue = temp
> >         return count
> > ```
> >
> > This bfs could be improved by the visited property. If we want to find out the shortest path, we only care about the first appearance of a node of a specific value. If the node value has already been added to the leave, there is no need to add the leave in a deeper layer. We could use the visited property to determine whether this value is appeared in the first time.
> >
> > ```python
> > class Solution:
> >     def numSquares(self, n: int) -> int:
> >         if n <= 1: return n
> >         maxFactor = int(sqrt(n))
> >         squares = []
> >         i = 1
> >         while i **2 <= n:
> >             squares.append(i ** 2)
> >             i += 1
> >         queue = {n}
> >         count = 0
> >         visited = [False] * (n + 1)
> >         while queue:
> >             count += 1
> >             temp = set()
> >             for number in queue:
> >                 for square in squares:
> >                     if square == number: return count
> >                     if square > number: break
> >                     if not visited[number - square]:
> >                         temp.add(number - square)
> >                         visited[number - square] = True
> >             queue = temp
> > ```
>
> Coin Change https://leetcode.com/problems/coin-change/
>
> > This problem is very similar to the perfect square which could also be solved by the bfs.
> >
> > > BFS
> > >
> > > ```python
> > > class Solution:
> > >     def coinChange(self, coins: List[int], amount: int) -> int:
> > >         if amount == 0: return 0
> > >         queue = [amount]
> > >         count = 0
> > >         coins = sorted(coins)
> > >         visited = [False] * (amount + 1)
> > >         while queue:
> > >             count += 1
> > >             temp = set()
> > >             for number in queue:
> > >                 for i, coin in enumerate(coins):
> > >                     if coin == number: 
> > >                         print(queue)
> > >                         return count
> > >                     if coin > number: 
> > >                         break
> > >                     left = number - coin
> > >                     if visited[left] == False:
> > >                         temp.add(number - coin)
> > >                         visited[left] = True
> > >             queue = temp
> > >         return -1
> > > ```
> > >
> > > DP: begin from smallest
> > >
> > > ```python
> > > class Solution:
> > >     def coinChange(self, coins: List[int], amount: int) -> int:
> > >         dp = [float('inf') for _ in range(amount + 1)]
> > >         dp[0] = 0
> > >         coins = sorted(coins)
> > >         minCoin = coins[0]
> > >         for i in range(minCoin, amount + 1):
> > >             for coin in coins:
> > >                 if coin > i: break
> > >                 dp[i] = min(dp[i], dp[i - coin] + 1)
> > >         return dp[amount] if dp[amount] < float('inf') else -1
> > > ```
> > >
> > > 
>
> 
>
> 