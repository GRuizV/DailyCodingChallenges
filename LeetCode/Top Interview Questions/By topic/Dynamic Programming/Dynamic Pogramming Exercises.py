'''
CHALLENGES INDEX

5. Longest Palindromic Substring (DP) (TP)
22. Generate Parentheses (DP) (BT)
53. Maximum Subarray (Array) (DQ) (DP)
55. Jump Game (Array) (DP) (GRE)
62. Unique Paths (DP)
70. Climbing Stairs (DP)
91. Decode Ways (DP)


*LL: Linked-Lists
*BS: Binary Search
*DP: Dynamic Programming
*RC: Recursion
*TP: Two-pointers
*FCD: Floyd's cycle detection (Hare & Tortoise approach)
*PS: Preffix-sum
*SW: Sliding-Window
*MEM: Memoization
*GRE: Greedy
*DQ: Divide and Conquer
*BT: Backtracking


(XX)

'''


'5. Longest Palindromic Substring'
# def x():

#     # Input
#     s = "cbbd"


#     # 1st Approach: Brute Force

#     # Creating the possible substrings from the input
#     subs = []

#     for i in range(1, len(s)+1):
        
#         for j in range((len(s)+1)-i):

#             subs.append(s[j:j+i])

#     # # validating
#     # print(subs)        

#     palindromes = sorted(filter(lambda x : True if x == x[::-1] else False, subs), key=len, reverse=True)

#     print(palindromes)

#     '''
#     Note: While the solution works, is evidently not efficient enough / Time Limit Exceeded.
#     '''

#     # 2nd Approach: Same brute force but less brute

#     max_len = 1
#     max_str = s[0]

#     for i in range(len(s)-1):

#         for j in range(i+1, len(s)):

#             sub = s[i:j+1]        

#             if (j-i)+1 > max_len and sub == sub[::-1]:

#                 max_len = (j-i)+1
#                 max_str = s[i:j+1]


#     print(max_str)

'22. Generate Parentheses'
# def x():

#     # Input
#     n = 3   # Expected Output: ['((()))', '(()())', '(())()', '()(())', '()()()']

#     # My Approach
#     def generateParenthesis(n):
    
#         if n == 1:
#             return ['()']

#         result = []

#         for i in generateParenthesis(n-1):
#             result.append('('+ i +')')
#             result.append('()'+ i )
#             result.append(i + '()')


#         return sorted(set(result))

#     print(generateParenthesis(4))

#     '''
#     Note: My solution kind of work but it was missing one variation, apparently with DFS is solved.
#     '''
#     # DFS Approach
#     def generateParenthesis(n):
        
#         res = []

#         def dfs (left, right, s):

#             if len(s) == 2*n:
#                 res.append(s)
#                 return

#             if left < n:
#                 dfs(left+1, right, s + '(')

#             if right < left:
#                 dfs(left, right+1, s + ')')

#         dfs(0,0,'')

#         return res

#     # Testing
#     print(generateParenthesis(4))

'53. Maximum Subarray'
# def x():

#     # Input

#     # Case 1
#     nums = [-2,1,-3,4,-1,2,1,-5,4]
#     # Output: 6 / [4,-1,2,1]

#     # Case 2
#     nums = [1]
#     # Output: 1 / [1]

#     # Case 3
#     nums = [5,4,-1,7,8]
#     # Output: 23 / [5,4,-1,7,8]
    
#     '''
#     My Approach

#     Intuition: Brute forcing
#         1. Store the sum of the max array
#         2. From the max len down to len = 1, evaluate with a sliding window each sub array.
#         3. Return the max sum
#     '''
#     def maxSubArray(nums):

#         max_sum = sum(nums)

#         # Handling the special case len = 1
#         if len(nums) == 1:
#             return max_sum
        
        
#         max_element = max(nums)
#         nums_len = len(nums)-1
#         idx = 0

#         while nums_len > 1:

#             if idx + nums_len > len(nums):
#                 nums_len -= 1
#                 idx = 0
#                 continue
                
#             sub_array = nums[idx:idx+nums_len]
#             sub_array_sum = sum(sub_array)

#             if sub_array_sum > max_sum:
#                 max_sum = sub_array_sum
                
#             idx += 1

#         # Handling the case where one element is greater than any subarray sum
#         if max_element > max_sum:
#             return max_element
        
#         return max_sum


#     print(maxSubArray(nums))
    
#     'Notes: My solution worked 94,7% of the cases, the time limit was reached.'


#     '''Kadane's Algorythm (for max subarray sum)'''

#     def maxSubArray(nums):

#         max_end_here = max_so_far = nums[0]

#         for num in nums[1:]:

#             max_end_here = max(num, max_end_here + num)
#             max_so_far = max(max_so_far, max_end_here)
        
#         return max_so_far


#     print(maxSubArray(nums))

#     'Notes: Apparently it was a classic problem'

'62. Unique Paths'
# def x():

#     '''
#     *1st Dynamic programming problem

#     Notes:

#     This problem was pretty similar to the one on Turing's tests, althought here is requested
#     to find a bigger scale of thar problem. The classic 'How many ways would be to get from x to y',

#     if the problem were only set to m = 2, it'd be solved with fibonacci, but sadly that was not the case,
#     here, Dynamic Programming was needed.

#     The problem is graphically explained here: https://www.youtube.com/watch?v=IlEsdxuD4lY

#     But the actual answer I rather take it from the leetCode's solutions wall, since is more intuitive to me.

#     '''

#     # Input
#     # Case 1:
#     m, n = 3, 7
#     # Output: 28

#     # Case 2:
#     m, n = 3, 2
#     # Output: 3

#     'Solution'
#     def uniquePaths(m: int, n: int) -> int:

#         # Handling the corner case in which any dimention is 0
#         if n == 0 or m == 0:
#             return 0


#         # Here the grid is initialized
#         result = [[0]*n for _ in range(m)]

#         # The first column of the grid is set to 1, since there is only (1) way to get to each cell of that column
#         for row in range(m):
#             result[row][0] = 1

#         # The first row of the grid is set to 1, since there is only (1) way to get to each cell of that row
#         for col in range(n):
#             result[0][col] = 1


#         # Here all the grid is traversed summing up the cells to the left and up, since are the only ways to get to the current cell
#         # The range starts in 1 since all the first column and row are populated, so the traversing should start in [1,1]
#         for i in range(1, m):

#             for j in range(1, n):

#                 result[i][j] = result[i-1][j] + result[i][j-1]
        

#         # The bottom right cell will store all the unique ways to get there
#         return result[-1][-1]

#     # Testing
#     print(uniquePaths(m, n))

'70. Climbing Stairs'
# def x():

#     # Input
#     # Case 1
#     n = 2
#     # Output = 2

#     # Case 2
#     n = 3
#     # Output = 3

#     # Case 2
#     n = 5
#     # Output = 3

#     'Solution'
#     def climbStairs(n:int) -> int:

#         res = [1,1]

#         for i in range(2, n+1):
#             res.append(res[i-2]+res[i-1])
        

#         return res[-1]

#     # Testing
#     print(climbStairs(n))

#     'Notes: The recursive solution, while elegant and eyecatching, is not as efficient as an iterative one'

'91. Decode Ways' 
# def x():
    
#     # Input
#     # Case 1:
#     s = '12'
#     # Output: 2

#     # Case 2:
#     s = '226'
#     # Output: 3

#     # Case 3:
#     s = '06'
#     # Output: 0

#     # Custom Case:
#     s = '112342126815'
#     # Output: 11


#     'My apporach'
#     # Auxiliary fibonacci pattern generator function
#     def fib(n):

#         res = [1,1]

#         for _ in range(n-1):
#             res.append(res[-2] + res[-1])
            
#         return res[1:]


#     def numDecodings(s:str) -> int:

#         if s[0] == '0':
#             return 0
        
#         if len(s) == 1:
#             return 1

#         substrings = []
#         subs = ''

#         if s[0] in ['1', '2']:
#             subs += s[0]

#         for i in range(1, len(s)+1):

#             if i == len(s):
#                 if subs != '':
#                     substrings.append(subs)

#             elif (s[i] in ['1', '2']) or (s[i-1] in ['1', '2'] and s[i] <= '6'):
#                 subs += s[i]

#             else:
#                 substrings.append(subs)
#                 subs = ''

#         cap = len(max(substrings, key=len))
#         possibilities = fib(cap)

#         res = 0

#         for i in substrings:

#             if i in '10' or '20':
#                 res += 1

#             else:
#                 res += possibilities[len(i)-1] 
        
#         return res

#     # Testing
#     print(numDecodings(s))

#     '''
#     Notes: 
#         This solution met 48% of expected results, there are a couple of cases I left unanalyzed.
#         Nevertheless, the logic of fibonaccying the parsing numbers works, perhaps with more time
#         a solution through this approach could work.
#     '''


#     'Dynamic Programming Approach'
#     def numDecodings(self, s):
        
#         dp = {len(s):1}

#         def backtrack(i):

#             if i in dp:
#                 return dp[i]

#             if s[i]=='0':
#                 return 0

#             if i==len(s)-1:
#                 return 1

#             res = backtrack(i+1)

#             if int(s[i:i+2])<27:
#                 res+=backtrack(i+2)
                
#             dp[i]=res

#             return res

#         return backtrack(0)











