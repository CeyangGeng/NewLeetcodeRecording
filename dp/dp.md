> ## The general solution for this is to think about how to apply the question to the base case, how to get the transfer function.

> 
>
> Minimum Path Sum  https://leetcode.com/problems/minimum-path-sum/
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
> > > If the coins are unlimited, then the inner loop should be incremental, otherwise, the inner loop should be decremental. 
> > >
> > > See the explanation here: https://leetcode.com/discuss/study-guide/1200320/Thief-with-a-knapsack-a-series-of-crimes.
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
> >
> > Ones and Zeros https://leetcode.com/problems/ones-and-zeroes/
> >
> > A list of 0-1 snapsack problem https://leetcode.com/discuss/study-guide/1200320/Thief-with-a-knapsack-a-series-of-crimes.
> >
> > > The base case in top-down method is different from the dp initilization part in bottom-up dp.
> > >
> > > In the two dimensional questions(start from the top left, end at the bottom right): the base case is start at the bottom right, the initialization in dp is top line and left column.
> > >
> > > In the one dimension question(start from the left and ends at right): the base case is start at the right most, the initialization is assign value to the left most.
> > >
> > > ```python
> > > # top-down method
> > > class Solution:
> > >     def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
> > >         memo = dict()
> > >         zeroOneCount = dict()
> > >         size = len(strs)
> > >         def helper(index, zc, oc, memo):
> > >             if index == size or (zc == 0 and oc == 0): return 0
> > >             if(index, zc, oc) in memo: return memo[(index, zc, oc)]
> > >             maxRes = 0
> > >             for i in range(index, size):
> > >                 string = strs[i]
> > >                 curZeroCount, curOneCount = string.count("0"), string.count("1")
> > >                 include = 0
> > >                 if curZeroCount <= zc and curOneCount <= oc:
> > >                     include = helper(i + 1, zc - curZeroCount, oc - curOneCount, memo) + 1
> > >                 exclude = helper(i + 1, zc, oc, memo)
> > >                 maxRes = max(maxRes, include, exclude)
> > >             memo[(index, zc, oc)] = maxRes
> > >             return maxRes
> > >         return helper(0, m, n, memo)
> > > ```
> > >
> > > ```python
> > > # bottom-up method
> > > class Solution:
> > >     def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
> > >         size = len(strs)
> > >         dp = [[[0 for _ in range(n + 1)]for _ in range(m + 1)] for _ in range(size + 1)]
> > >         for k in range(1, size + 1):
> > >             string = strs[k - 1]
> > >             zeroCount, oneCount = string.count("0"), string.count("1")
> > >             for i in range( m + 1):
> > >                 for j in range( n + 1):
> > >                     if zeroCount <= i and oneCount <= j:
> > >                         dp[k][i][j] = max(dp[k - 1][i][j], dp[k - 1][i - zeroCount][j - oneCount] + 1)
> > >                     else:
> > >                         dp[k][i][j] = dp[k - 1][i][j]
> > >         return dp[size][m][n]
> > > ```
> >
> > Partition Equal Subset Sum https://leetcode.com/problems/partition-equal-subset-sum/
> >
> > > ```python
> > > # classic snapsack problem 
> > > # pick or not
> > > class Solution:
> > >  def canPartition(self, nums: List[int]) -> bool:
> > >      total = sum(nums)
> > >      if total & 1 == 1: return False
> > >      target = int(total / 2)
> > >      size = len(nums)
> > >      dp =[ [False for _ in range(target + 1)] for _ in range(size + 1)]
> > >      dp[0][0] = True
> > >      for i in range(1, size + 1):
> > >          dp[i][0] = True
> > >      for j in range(1, target + 1):
> > >          dp[0][j] = False
> > >      for i in range(1, size + 1):
> > >          for j in range(1, target + 1):
> > >              dp[i][j] = dp[i - 1][j]
> > >              if j >= nums[i - 1]:
> > >                  dp[i][j] = dp[i][j] or dp[i - 1][j - nums[i - 1]]
> > >      return dp[size][target]
> > > ```
> > >
> > > ```python
> > > # space optimization
> > > # if the element is limited, the outer loop should be the element, the inner loop should be the target and decremental.
> > > class Solution:
> > >     def canPartition(self, nums: List[int]) -> bool:
> > >         total = sum(nums)
> > >         if total & 1 == 1: return False
> > >         target = int(total / 2) 
> > >         dp = [False for _ in range(target + 1)]
> > >         dp[0] = True
> > >         for num in nums:
> > >             for i in range(target, num - 1, -1):
> > >                 dp[i] = dp[i] or dp[i - num]
> > >         return dp[-1]
> > > ```
> >
> >  Integer Break https://leetcode.com/problems/integer-break/
> >
> > > ```python
> > > # mathematical solution
> > > class Solution:
> > >     def integerBreak(self, n: int) -> int:
> > >         if n == 2: return 1
> > >         if n == 3: return 2
> > >         product = 1
> > >         while(n > 4):
> > >             product *= 3
> > >             n -= 3
> > >         product *= n
> > >         return product
> > > ```
> > >
> > > ```python
> > > # Normal dp solution
> > > # the outer loop traverse all the sum factors
> > > # the inner loop traverse all the sums from factor to the target
> > > # If we don't pick factor i, then leave dp[j]
> > > # If we pick factor, then there are two possibilities: use j - i itself or use dp[j - 1]
> > > class Solution:
> > >     def integerBreak(self, n: int) -> int:
> > >         dp = [1] * (n + 1)
> > >         for i in range(2, n):
> > >             for j in range(i + 1, n + 1):
> > >                 dp[j] = max(dp[j], dp[j - i] * i, (j - i) * i)
> > >         return dp[-1]
> > > ```
> >
> > Coin Change 2 https://leetcode.com/problems/coin-change-2/
> >
> > > ```python
> > > # To differenciate all the combinations, we require that the combinations must ends with the number num to add up to a certain sum.
> > > class Solution:
> > >     def change(self, amount: int, coins: List[int]) -> int:
> > >         dp = [0] * (amount + 1)
> > >         dp[0] = 1
> > >         for coin in coins:
> > >             for i in range(coin, amount + 1):
> > >                 dp[i] += dp[i - coin]
> > >         return dp[-1]
> > > ```
> >
> > order-1:
> >
> > 
> >
> > ```python
> > for each sum in dp[]
> >     for each num in nums[]
> >         if (sum >= num)
> >             dp[sum] += dp[sum-num];
> > ```
> >
> > 
> >
> > order-2:
> >
> > 
> >
> > ```python
> > for each num in nums[]
> >     for each sum in dp[]  >= num
> >         dp[sum] += dp[sum-num];
> > ```
> >
> > order-1 is used to calculate the number of combinations considering different sequences
> > order-2 is used to calculate the number of combinations NOT considering different sequences
> >
> > 
> >
> > Combination Sum IV. https://leetcode.com/problems/combination-sum-iv/
> >
> > > ```python
> > > class Solution:
> > >     def combinationSum4(self, nums: List[int], target: int) -> int:
> > >         dp = [0] * (target + 1)
> > >         dp[0] = 1
> > >         for i in range(1, target + 1):
> > >             for num in nums:
> > >                 if num <= i:
> > >                     dp[i] += dp[i - num]
> > >         return dp[-1]
> > > ```
> >
> > 2 Keys Keyboard https://leetcode.com/problems/2-keys-keyboard/
> >
> > > ```python
> > > class Solution:
> > >     def minSteps(self, n: int) -> int:
> > >         dp = [i for i in range(n + 1)]
> > >         dp[1] = 0
> > >         for i in range(2, n + 1):
> > >             for j in range(int(i / 2), 1, -1):
> > >                 if (i % j) == 0:
> > >                     dp[i] = dp[j] + int(i / j)
> > >                     break
> > >         return dp[-1]
> > > ```
> >
> > Min Cost Climbing Stairs https://leetcode.com/problems/min-cost-climbing-stairs/
> >
> > ```python
> > class Solution:
> >     def minCostClimbingStairs(self, cost: List[int]) -> int:
> >         cost.append(0)
> >         n = len(cost)
> >         for i in range(2, n):
> >             cost[i] += min(cost[i - 1], cost[i - 2])
> >         return cost[-1]
> > ```
> >
> > 
>
> > snapsack problem solution thought:
> >
> > If for each element in the given list, the user could choose to use it or not, then it is a snapsack problem.
> >
> > Typically, there are always two level nested for loop. 
> >
> > > - If the order of the combinations is not considered, then the outer loop should be the elements, the inner loop should be the index of the dp list.
> > >
> > > - If the order of the combinations is considered, then the outer loop should be the index of the dp list, the inner loop should be the elements.
> >
> > There are incremental and decremental order of the element inner loop. 
> >
> > > - If the elements are infinite, then the order should be incremental. 
> > > - If the elements are decremental, then the order should be decremental.
> >
> >  Minimum Number of Refueling Stops https://leetcode.com/problems/minimum-number-of-refueling-stops/
> >
> > > ```python
> > > # snapsack
> > > class Solution:
> > >     def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
> > >         size = len(stations)
> > >         dp = [startFuel] + [-1] * size
> > >         for i in range(size):
> > >             for j in range(i + 1, 0, -1):
> > >                 station = stations[i]
> > >                 dis, gas = station[0], station[1]
> > >                 if dp[j - 1] >= dis:
> > >                     dp[j] = max(dp[j], dp[j - 1] + gas)
> > >         for i in range(size + 1):
> > >             if dp[i] >= target: return i
> > >         return -1
> > > ```
> > >
> > > ```python
> > > # greedy
> > > class Solution:
> > >     def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
> > >         size = len(stations)
> > >         i, res = 0, 0
> > >         cur = startFuel
> > >         queue = []
> > >         while cur < target:
> > >             while i < size and stations[i][0] <= cur:
> > >                 heapq.heappush(queue, -stations[i][1])
> > >                 i += 1
> > >             if  not queue: return -1
> > >             cur += (-heapq.heappop(queue))
> > >             res += 1
> > >         return res
> > > ```
> >
> > Minimum Falling Path Sum https://leetcode.com/problems/minimum-falling-path-sum/
> >
> > > ```python
> > > class Solution:
> > >     def minFallingPathSum(self, matrix: List[List[int]]) -> int:
> > >         m, n = len(matrix), len(matrix[0])
> > >         for i in range(1, m):
> > >             for j in range(n):
> > >                 if j == 0: matrix[i][j] += min(matrix[i - 1][j], matrix[i - 1][j + 1])
> > >                 elif j == n -1: matrix[i][j] += min(matrix[i - 1][j - 1], matrix[i - 1][j])
> > >                 else: matrix[i][j] += min(matrix[i - 1][j - 1], matrix[i - 1][j], matrix[i - 1][j + 1])
> > >         return min(matrix[-1])
> > > ```
> >
> > Minimum Cost For Tickets
> >
> > > ```python
> > > class Solution:
> > >     def mincostTickets(self, days: List[int], costs: List[int]) -> int:
> > >         maxDay = days[-1]
> > >         dp = [0] + [float("inf")] * maxDay
> > >         for i in range(1, maxDay + 1):
> > >             dp[i] = dp[i - 1]
> > >             if i in days:
> > >                 oneDayBeforeIndex = max(0, i - 1)
> > >                 sevenDayBeforeIndex = max(0, i - 7)
> > >                 thirtyDayBeforeIndex = max(0, i - 30)
> > >                 dp[i] = min(dp[oneDayBeforeIndex] + costs[0], dp[sevenDayBeforeIndex] + costs[1], dp[thirtyDayBeforeIndex] + costs[2])
> > >         return dp[-1]
> > > ```
> >
> >  Last Stone Weight II
> >
> > > ```python
> > > class Solution:
> > >     def lastStoneWeightII(self, stones: List[int]) -> int:
> > >         dp = {0}
> > >         maxTotal = 0
> > >         for stone in stones:
> > >             maxTotal += stone
> > >             for total in range(maxTotal, stone - 1, -1):
> > >                 if total - stone in dp:
> > >                     dp.add(total)
> > >         half = maxTotal / 2
> > >         minDiff = float('inf')
> > >         for i in dp:
> > >             minDiff = min(minDiff, abs(maxTotal - 2 * i))
> > >         return minDiff
> > > ```
> >
> > Unique Paths II
> >
> > > ```python
> > > class Solution:
> > >     def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
> > >         if obstacleGrid[0][0] == 1: return 0
> > >         m, n = len(obstacleGrid), len(obstacleGrid[0])
> > >         dp = [[0 for _ in range(n)] for _ in range(m)]
> > >         dp[0][0] = 1
> > >         for j in range(1, n):
> > >             if obstacleGrid[0][j] == 1: break
> > >             dp[0][j] = 1
> > >         for i in range(1, m):
> > >             if obstacleGrid[i][0] == 1: break
> > >             dp[i][0] = 1
> > >         for i in range(1, m):
> > >             for j in range(1, n):
> > >                 dp[i][j] = 0 if obstacleGrid[i][j] == 1 else dp[i - 1][j] + dp[i][j - 1]
> > >         return dp[-1][-1]
> > > ```
> >
> >  Partition Equal Subset Sum
> >
> > > ```python
> > > class Solution:
> > >     def canPartition(self, nums: List[int]) -> bool:
> > >         total = sum(nums)
> > >         if(total & 1 == 1): return False
> > >         total = int(total / 2)
> > >         dp = [False] * (total + 1)
> > >         dp[0] = True
> > >         for num in nums:
> > >             for s in range(total, num - 1, -1):
> > >                 dp[s] |= dp[s - num]
> > >         print(dp)
> > >         return dp[-1]
> > > ```
> >
> > Target Sum https://leetcode.com/problems/target-sum/
> >
> > > ```python
> > > class Solution:
> > >     def findTargetSumWays(self, nums: List[int], target: int) -> int:
> > >         total = sum(nums)
> > >         tmp = (total + target)
> > >         if(tmp & 1 == 1): return 0
> > >         partialTotal = int(tmp / 2)
> > >         dp = [1] + [0] * partialTotal
> > >         for num in nums:
> > >             for i in range(partialTotal, num-1, -1):
> > >                 dp[i] += dp[i - num]
> > >         return dp[-1]
> > > ```
> >
> > Out of Boundary Paths https://leetcode.com/problems/out-of-boundary-paths/
> >
> > > ```python
> > > class Solution:
> > >     def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
> > >         memo = dict()
> > >         def helper(row, col, leftMove):
> > >             if row in {-1, m} or col in {-1, n}:
> > >                 if leftMove >= 0: return 1
> > >                 else: return 0
> > >             if leftMove == 0: return 0
> > >             if (row, col, leftMove) in memo: return memo[(row, col,leftMove)]
> > >             leftMoves = leftMove - 1
> > >             res = 0
> > >             res += helper(row - 1, col, leftMoves)
> > >             res += helper(row + 1, col, leftMoves)
> > >             res += helper(row, col - 1, leftMoves)
> > >             res += helper(row, col + 1, leftMoves)
> > >             memo[(row, col, leftMove)] = res
> > >             return res 
> > >         return helper(startRow, startColumn, maxMove) % (10 ** 9 + 7)
> > > ```
> >
> > 
> >
> > 
> >
> > 
>
> 