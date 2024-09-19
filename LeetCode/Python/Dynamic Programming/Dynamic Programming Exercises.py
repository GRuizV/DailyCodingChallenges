'''
CHALLENGES INDEX

5. Longest Palindromic Substring (DP) (TP)
22. Generate Parentheses (DP) (BT)
53. Maximum Subarray (Array) (DQ) (DP)
55. Jump Game (Array) (DP) (GRE)
62. Unique Paths (DP)
70. Climbing Stairs (DP)
91. Decode Ways (DP)
118. Pascal's Triangle (Array) (DP)
121. Best Time to Buy and Sell Stock (Array) (DP)
122. Best Time to Buy and Sell Stock II (Array) (DP) (GRE)
124. Binary Tree Maximum Path Sum (DP) (Tree) (DFS)
131. Palindrome Partitioning (DP) (BT)
139. Word Break (DP)
140. Word Break II (DP) (BT)
152. Maximum Product Subarray (Array) (DP)
198. House Robber (Array) (DP)
279. Perfect Squares (DP)
300. Longest Increasing Subsequence (Array) (DP)
322. Coin Change (DP)

32. Longest Valid Parentheses (Stack) (DP)
64. Minimum Path Sum (Matrix) (DP)


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
*BFS & DFS: Breadth-First Search & Depth-First Search
*Arrays, Hash Tables & Matrices
*Sorting
*Heaps, Stacks & Queues
*Graphs & Trees
*Others


(22)
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

'55. Jump Game'
# def x():

#     # Input
#     # Case 1
#     nums = [2,3,1,1,4]
#     # Output: True

#     # Case 2
#     nums = [3,2,1,0,4]
#     # Output: False

#     # Custom Case
#     nums = [2,0,0]
#     # Output: 

#     '''
#     My Approach

#         Intuition - Brute Force:

#             - I will check item by item to determine if the end of the list is reachable

#     '''
#     def canJump(nums:list[int]) -> bool:

#         # Corner case: nums.lenght = 1 / nums[0] = 0
#         if len(nums) == 1 and nums[0] == 0:
#             return True
        
#         idx = 0

#         while True:

#             idx += nums[idx]

#             if idx >= len(nums)-1:
#                 return True
            
#             if nums[idx] == 0 and idx < len(nums)-1:
#                 return False
            
#     # Testing
#     print(canJump(nums))

#     'Notes: This solution suffice 91,2% of the case'


#     'Backtrack Approach'
#     def canJump(nums: list[int]) -> bool:

#         # Handle corner case: One-element input
#         if len(nums) == 1:
#             return True  

#         # Start at num[-2] since nums[-1] is given
#         backtrack_index = len(nums)-2 

#         # At nums[-2] we only need to jump 1 to get to nums[-1]
#         jump = 1  

#         # Iterate through the element backwards
#         while backtrack_index > 0:

#             # If we can get to the nearest lily pad
#             if nums[backtrack_index] >= jump:

#                 #now we have a new nearest lily pad
#                 jump = 1 

#             # Else the jump is one bigger than before
#             else:               
#                 jump += 1 

#             # Move one backwards
#             backtrack_index -= 1
        
#         #Now that we know the nearest jump to nums[0], we can finish
#         if jump <= nums[0]: 
#             return True
        
#         else:
#             return False 

#     'Notes: Right now I am not that interested in learning bactktracking, that will be for later'

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

'''118. Pascal's Triangle'''
# def x():

#     # Input
#     # Case 1
#     numRows = 5
#     #Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]

#     # Case 2
#     numRows = 1
#     #Output: [[1]]
      

#     '''
#     My Approach

#         Intuition:
#             initialize a preset solution to [[1],[1,1]] and according to the
#             parameter passed in the function, start to sum and populate this sums to a list
#             like [1]+[resulting_sums]+[1] and return that back to the preset solution, to operate over that
#             new element,

#                 The number of loops will be numRows - 2 (given the 2 initial elements)
#     '''

#     def generate(numRows:int) -> list[list[int]]:

#         result = [[1],[1,1]]

#         if numRows == 1:
#             return [result[0]]
        
#         if numRows == 2:
#             return result
        

#         for i in range(1, numRows-1):

#             new_element = []

#             for j in range(i):
#                 new_element.append(result[-1][j]+result[-1][j+1])

#             if new_element:
#                 result.append([1]+new_element+[1])

#         return result

#     # Testing
#     print(generate(numRows=5))

#     'It worked!'

'''121. Best Time to Buy and Sell Stock'''
# def x():

#     # Input
#     #Case 1
#     prices = [7,1,5,3,6,4]
#     #Output: 5

#     #Case 2
#     prices = [7,6,4,3,1]
#     #Output: 0


#     '''
#     My approach
#         Intuition
#             - Corner Case: if is a ascendingly sorted list, return 0.
            
#             - Pick the first item and set the profit as the max between the current profit and the difference between the first element
#             the max value from that item forward.
            
#             Do this in a while loop until len(prices) = 1.
#     '''

#     def maxProfit(prices: list[int]) -> int:

#         profit = 0

#         if prices == sorted(prices, reverse=True):
#             return profit        

#         while len(prices) > 1:

#             purchase = prices.pop(0)
#             profit = max(profit, max(prices)-purchase)
        
#         return profit

#     # Testing
#     print(maxProfit(prices=prices))

#     'This approach met 94% of the results'


#     '''Kadane's Algorithm'''
#     def maxProfit(prices: list[int]) -> int:

#         buy = prices[0]
#         profit = 0

#         for num in prices[1:]:

#             if num < buy:
#                 buy = num
            
#             elif num-buy > profit:
#                 profit = num - buy
        
        
#         return profit

#     # Testing
#     print(maxProfit(prices=prices))

#     'Done'

'''122. Best Time to Buy and Sell Stock II'''
# def x():

#     #Input
#     #Case 1
#     prices = [7,1,5,3,6,4]
#     #Output: 7

#     #Case 2
#     prices = [1,2,3,4,5]
#     #Output: 4

#     #Case 3
#     prices = [7,6,4,3,1]
#     #Output: 0

#     #Custom Case
#     prices = [3,3,5,0,0,3,1,4]
#     #Output: 0


#     'My approach'
#     def maxProfit(prices:list[int]) -> int:

#         if prices == sorted(prices, reverse=True):
#             return 0
        
#         buy = prices[0]
#         buy2 = None
#         profit1 = 0
#         profit2 = 0
#         total_profit = 0

#         for i in range(1, len(prices)):

#             if prices[i] < buy:
#                 buy = prices[i]
            
#             elif prices[i] - buy >= profit1:            
#                 profit1 = prices[i] - buy
#                 buy2 = prices[i] 

#                 for j in range(i+1, len(prices)):

#                     if prices[j] < buy2:
#                         buy2 = prices[j]

#                     elif prices[j] - buy2 >= profit2:
#                         profit2 = prices[j] - buy2
#                         total_profit = max(total_profit, profit1 + profit2)
            
#             total_profit = max(total_profit, profit1)

#         return total_profit

#     # Testing
#     print(maxProfit(prices=prices))

#     'This solution went up to solve 83% of the cases, the gap was due to my lack of understanding of the problem'


#     '''Same Kadane's but modified'''
#     def maxProfit(prices:list[int]) -> int:

#         max = 0 
#         start = prices[0]
#         len1 = len(prices)

#         for i in range(0 , len1):

#             if start < prices[i]: 
#                 max += prices[i] - start

#             start = prices[i]

#         return max

#     # Testing
#     print(maxProfit(prices=prices))

#     'My mistake was to assume it can only be 2 purchases in the term, when it could be as many as it made sense'

'''124. Binary Tree Maximum Path Sum'''
# def x():

#     # Base 
#     class TreeNode(object):
#         def __init__(self, val=0, left=None, right=None):
#             self.val = val
#             self.left = left
#             self.right = right

#     # Input
#     #Case 1
#     tree_layout = [1,2,3]
#     root = TreeNode(val=1, left=TreeNode(val=2), right=TreeNode(val=3))
#     #Output: 6

#     #Case 2
#     tree_layout = [-10,9,20,None, None,15,7]
#     left = TreeNode(val=9)
#     right = TreeNode(val=20, left=TreeNode(val=15), right=TreeNode(val=7))
#     root = TreeNode(val=-10, left=left, right=right)
#     #Output: 42

#     #Custom Case
#     tree_layout = [1,-2,3,1,-1,-2,-3]
#     left = TreeNode(val=-2, left=TreeNode(val=1), right=TreeNode(val=3))
#     right = TreeNode(val=-3, left=TreeNode(val=-2, left=TreeNode(val=-1)))
#     root = TreeNode(val=1, left=left, right=right)
#     #Output: 3


#     '''
#     My Approach

#         Intuition:
#             - Make a preorder traversal tree list.
#             - Apply Kadane's algorithm to that list.
#     '''

#     def maxPathSum(root:TreeNode) -> int:

#         #First, Preorder
#         path = []

#         def preorder(node:TreeNode) -> None:

#             if node:
#                 preorder(node=node.left)
#                 path.append(node.val)
#                 preorder(node=node.right)

#         preorder(node=root)

#         #Now Kadane's
#         max_so_far = max_end_here = path[0]

#         for num in path[1:]:

#             max_end_here = max(num, max_end_here + num)
#             max_so_far = max(max_so_far, max_end_here)

#         return max_so_far

#     # Testing
#     print(maxPathSum(root=root))

#     '''
#     Notes:
#         - On the first run it went up to 59% of the cases, thats Kudos for me! :D
#         - The problem with this algorithm is that it supposes that after reaching a parent and child node,
#         it's possible to go from a right child to the parent of the parent and that either forcibly makes
#         to pass twice from the parent before going to the granparent, or that one grandchild is connected
#         to the grandfather, which is also out of the rules.

#         I misinterpret this because one of the examples showed a path [leftchild, parent, rightchild] which
#         is valid only if we don't want to pass thruough the grandparent.
        
#         The best choice here is to make a recursive proning algorithm
#     '''


#     'A recursive approach'
#     def maxPathSum(root):

#         max_path = float('-inf') #Placeholder

#         def get_max_gain(node):

#             nonlocal max_path

#             if not node:
#                 return 0
            
#             gain_on_left = max(get_max_gain(node.left),0)
#             gain_on_right = max(get_max_gain(node.right),0)

#             current_max_path = node.val + gain_on_left + gain_on_right
#             max_path = max(max_path, current_max_path)

#             return node.val + max(gain_on_left, gain_on_right)
        
#         get_max_gain(root)

#         return max_path

#     # Testing
#     print(maxPathSum(root))

#     'Done'

'''131. Palindrome Partitioning'''
# def x():

#     # Input
#     # Case 1
#     s = 'aab'
#     # Output: [["a","a","b"],["aa","b"]]

#     # Custom Case
#     s = 'aabcdededcbada'
#     # Output: [["a","a","b"],["aa","b"]]

#     # Custom Case
#     s = 'aabcdededcbada'
#     # Output: ['abcdededcba', 'bcdededcb', 'cdededc', 'deded', 'ded', 'ede', 'ded', 'ada', 'aa'] 


#     '''
#     My Approach

#         Intuition:

#             Here I don't actually have much ideas in how to solve it, but one good approach
#             I think woul dbe to make a function that can pull all the palindroms present in a string.

#             that could be a good start point.
#     '''

#     def palindromes(string:str) -> list[str]:

#         s_len = len(string)
#         palindromes = []

#         for i in range(s_len, 1, -1):   # from s_len down to length 2 of substring
        
#             j = 0

#             while j + i <= s_len: 

#                 subs = string[j:j+i]

#                 if subs == subs[::-1]:

#                     palindromes.append(subs)

#                 j += 1

#         print(palindromes)

#     # Testing
#     # Printout: ['abcdededcba', 'bcdededcb', 'cdededc', 'deded', 'ded', 'ede', 'ded', 'ada', 'aa'] 
#     palindromes(string=s)

#     '''
#     Notes: At least this I was able to do, from here on, I am feeling I am going to brute forcing this and it won't end up being efficient.

#         I didn't actually solved it but I don't want to waste more time over this.
#     '''

'''139. Word Break'''
# def x():

#     #Input
#     #Case 1
#     s = "leetcode" 
#     wordDict = ["leet","code"]
#     #Output: True

#     #Case 2
#     s = "applepenapple"
#     wordDict = ["apple","pen"]
#     #Output: True

#     #Case 3
#     s = "catsandog"
#     wordDict = ["cats","dog","sand","and","cat"]
#     #Output: False


#     '''
#     My Approach

#         Intuition (Brute-force):

#             in a while loop go word for word in the dict checking if the word exists in the string:

#                 - If it does: Overwrite the string taking out the found word / else: go to the next word

#                 The loop will be when either no words are found in the string or the string is empty

#                 if after the loop the string is empty, return True, otherwise False
#     '''

#     def workBreak(string:str, word_dict:list[str]) -> bool:

#         j = 0

#         while j < len(word_dict):

#             if word_dict[j] in string:

#                 w_len = len(word_dict[j])
#                 idx = string.find(word_dict[j])
#                 string = string[:idx]+string[idx+w_len:]

#                 j = 0
        
#             else:
#                 j += 1

#         return False if string else True

#     print(workBreak(string=s, word_dict=wordDict))

#     'Note: This solution goes up to the 74% of the test cases'


#     'Dynamic Programming Approach'
#     def workBreak(string:str, word_dict:list[str]) -> bool:

#         dp = [False] * (len(s) + 1) # dp[i] means s[:i+1] can be segmented into words in the wordDicts 
#         dp[0] = True

#         for i in range(len(s)):

#             for j in range(i, len(s)):
                
#                 i_dp = dp[i]
#                 sub_s = s[i: j+1]
#                 test = sub_s in wordDict

#                 if i_dp and test:
#                     dp[j+1] = True
                    
#         return dp[-1]

#     print(workBreak(string=s, word_dict=wordDict))

#     'Done'

'''140. Word Break II'''
# def x():

#     #Input
#     #Case 1
#     s = "catsanddog"
#     wordDict = ["cat","cats","and","sand","dog"]
#     #Output: ["cats and dog","cat sand dog"]

#     #Case 2
#     s = "pineapplepenapple"
#     wordDict = ["apple","pen","applepen","pine","pineapple"]
#     #Output: ["pine apple pen apple","pineapple pen apple","pine applepen apple"]

#     #Case 3
#     s = "catsandog"
#     wordDict = ["cats","dog","sand","and","cat"]
#     #Output: []


#     '''
#     My Approach

#         Intuition:

#             - With the solution of the last exercise, bring the found words into a list and join them to from a sentence.
#             - In a loop, check if the first found word is the same of the last sentece, if do, keep searching for another word,
#                 - if not found words after looping from the first character, end the loop.
#     '''

#     def wordBreak(s:str, wordDict:list[str]) -> list[str]:

#         sentences = []
#         sent = []
#         string = s
#         lasts_first_word = []

#         while True:

#             j = 0

#             while j < len(string):

#                 if string[0:j+1] in wordDict and string[0:j+1] not in lasts_first_word:

#                     sent.append(string[0:j+1])
#                     string = string[j+1:]
#                     j = 0
                
#                 else:
#                     j += 1
            

#             if sent:
#                 sentences.append(' '.join(sent))
#                 string = s
#                 lasts_first_word.append(sent[0])
#                 sent = []
            
#             else:
#                 break
        
#         return sentences        

#     # Testing
#     print(wordBreak(s=s, wordDict=wordDict))

#     "Note:This solution doesn't even get to pass all the initial test cases, but at least it worked as a challenge to do at least one"


#     'Backtracking & Recursion approach'
#     def wordBreakHelper(s:str, start:int, word_set:set, memo:dict) -> list[str]:

#         if start in memo:
#             return memo[start]
        
#         valid_substr = []

#         if start == len(s):
#             valid_substr.append('')

#         for end in range(start+1, len(s)+1):

#             prefix = s[start:end]

#             if prefix in word_set:

#                 suffixes = wordBreakHelper(s, end, word_set, memo)

#                 for suffix in suffixes:

#                     valid_substr.append(prefix + ('' if suffix == '' else ' ') + suffix)

#         memo[start] = valid_substr

#         return valid_substr
            

#     def wordBreak(s:str, wordDict: list[str]) -> list[str]:

#         memo = {}
#         word_set = set(wordDict)
#         return wordBreakHelper(s, 0, word_set, memo)

#     # Testing
#     print(wordBreak(s=s, wordDict=wordDict))

#     'Done'

'''152. Maximum Product Subarray'''
# def x():

#     # Input
#     # Case 1
#     input = [2,3,-2,4]
#     # Output: 6 / [2,3] has the largest product

#     # Case 2
#     input = [-2,0,-1]
#     # Output: 0 / all products are 0

#     # Custom Case
#     input = [-2,3,-4]
#     # Output: 0 / all products are 0


#     '''
#     My approach

#         Intuition

#             This is a variation of Kadane's Algorithm, and may be solve same way
#             as the original
#     '''

#     def maxProduct(nums:list[int]) -> int:

#         if len(nums) == 1:
#             return nums[0]

#         max_ends_here, max_so_far = nums[0]

#         for num in nums[1:]:
        
#             max_ends_here = max(num, max_ends_here * num)
#             max_so_far = max(max_so_far, max_ends_here)

#         return max_so_far

#     # Testing
#     print(maxProduct(nums=input))

#     '''
#     Original Kadane's modified to compute product solved 51% of the cases. 
#     But, apparently with capturing the min_so_far and having a buffer to hold the max_so_far to not interfere with the
#         min_so_far calculation, the problem is solved
#     '''

#     '''Another Kadane's Mod. Approach'''
#     def maxProduct(nums:list[int]) -> int:

#         if len(nums) == 1:
#             return nums[0]

#         max_so_far = min_so_far = result = nums[0]

#         for num in nums[1:]:
        
#             temp_max = max(num, max_so_far * num, min_so_far * num)
#             min_so_far = min(num, max_so_far * num, min_so_far * num)
#             max_so_far = temp_max

#             result = max(result, max_so_far)

#         return result

#     # Testing
#     print(maxProduct(nums=input))

#     'Done'

'''198. House Robber'''
# def x():

#     # Input
#     # Case 1
#     nums = [1,2,3,1]
#     # Output: 4

#     # Case 2
#     nums = [2,7,9,3,1]
#     # Output: 12

#     # Custom Case
#     nums = [2,1,1,2]
#     # Output: 12


#     'DP Approach / space: O(n)'
#     def rob(nums: list[int]) -> int:
        
#         # Handling corner cases
#         if len(nums) == 1:
#             return nums[0]
        
#         # Initializing the aux array
#         dp = [0] * len(nums)
#         dp[0] = nums[0]
#         dp[1] = max(dp[0], nums[1])

#         for i in range(2, len(nums)):

#             dp[i] = max(dp[i-1], dp[i-2] + nums[i])

#         return dp[-1]

#     # Testing
#     print(rob(nums=nums))
                    
#     'Note: This could be done in O(1) space'

        
#     'DS Approach / space: O(1)'
#     def rob(nums: list[int]) -> int:
        
#         # Handling corner cases
#         if len(nums) == 1:
#             return nums[0]
        
#         # Initializing the aux array
#         prev_rob = 0
#         max_rob = 0

#         for num in nums:

#             temp = max(max_rob, prev_rob + num)
#             prev_rob = max_rob
#             max_rob = temp
        
#         return max_rob

#     # Testing
#     print(rob(nums=nums))

#     'Done'

'''279. Perfect Squares'''
# def x():

#     # Input
#     # Case 1
#     n = 12
#     # Output: 3 (4+4+4)

#     # Case 2
#     n = 13
#     # Output: 2 (4+9)

#     # Custom case
#     n = 43
#     # Output: 3 

#     # Custom case
#     n = 67
#     # Output: 3 


#     '''
#     My Approach

#         Intuition:
#             - Build the possible addends (Each number that its 2nd power is less than n).
#             - Reverse the addends (To have them from bigger to smaller).
#             - Iteratively check from bigger to smaller how many addends can be grouped to sum up using modulo and division.
#                 + If a division if the group is still short reach the n, go to the next addend to fill up.
#             - When the group is completed, start the process over but starting from the next addend.
#                 The last group will always be the largest, since it consists of a groups of 1's.
#             - Return the count of the shortest group.
#     '''

#     def num_squares(n:int) -> int:

#         # Define the holder of the groups
#         result = []

#         # Define the holder and the indext to populate the addends
#         addends = []
#         i = 1

#         # Populate the addends / ex: [1, 4, 9]
#         while i*i <= n:
#             addends.append(i*i)
#             i += 1

#         # Reverse the addends
#         addends = addends[::-1]

#         # Form the groups
#         for i in range(len(addends)):

#             group = []

#             for j in range(i, len(addends)):

#                 if sum(group) == n:
#                     break
            
#                 if (n-sum(group))/addends[j] >= 1:                
#                     group += ([addends[j]] * ((n-sum(group))//addends[j]))
            
#             result.append(group) if len(group) != n else None

#         # Sort the groups from the shortest to the largest
#         result.sort(key=len)

#         #return the shortest
#         return len(result[0])

#     # Testing
#     print(num_squares(n=n))

#     'This solution cleared 96% of the test cases, the actual DP solution didnt made sense to me'

'''300. Longest Increasing Subsequence'''
# def x():

#     # Input
#     # Case 1
#     nums = [10,9,2,5,3,7,101,18]
#     # Output: 4

#     # Case 2
#     nums = [0,1,0,3,2,3]
#     # Output: 4

#     # Case 3
#     nums = nums = [7,7,7,7,7,7,7]
#     # Output: 1


#     'DP Solution'
#     def lengthOfLIS(nums: list[int]) -> int:    
        
#         # Handle corner case
#         if not nums:
#             return 0        

#         # Initialize the dp array
#         dp = [1] * len(nums)

#         # Iterate through the elements of the list, starting from the second
#         for i in range(1, len(nums)):

#             for j in range(i):

#                 if nums[i] > nums[j]:
#                     dp[i] = max(dp[i], dp[j]+1)

#         return max(dp)

#     'Done'

'''322. Coin Change'''
# def x():

#     # Input
#     # Case 1
#     coins = [1,2,5]
#     amount = 11
#     # Output: 3

#     # Case 2
#     coins = [2]
#     amount = 3
#     # Output: -1

#     # Custome Case
#     coins = [186,419,83,408]
#     amount = 6249
#     # Output: 20


#     'My Approach (Greedy approach)'
#     def coin_change(coins:list[int], amount: int) -> int:

#         # Handle Corner Case
#         if not amount:
#             return 0
        
#         # Sort the coins decreasingly
#         coins = sorted(coins, reverse=True)

#         # Initialize the coins counter
#         result = 0

#         # Iterate through
#         for coin in coins:

#             if coin <= amount:

#                 result += amount // coin
#                 amount %= coin
            
#             if not amount:
#                 return result
        
#         # If the execution get to this point, it means it was not an exact number of coins for the total of the amount
#         return -1

#     # Testing
#     print(coin_change(coins=coins, amount=amount))

#     'Note: This is a Greedy approach and only met up 27% of test cases'


#     'DP Approach'
#     def coin_change(coins:list[int], amount: int) -> int:

#         # DP INITIALIZATION
#         # Initialize the dp array
#         dp = [float('inf')] * (amount+1)

#         # Initialize the base case: 0 coins for amount 0
#         dp[0] = 0

#         # DP TRANSITION
#         for coin in coins:

#             for x in range(coin, amount+1):
#                 dp[x] = min(dp[x], dp[x-coin] + 1)

#         # Return result
#         return dp[amount] if dp[amount] != float('inf') else -1

#     # Testing
#     print(coin_change(coins=coins, amount=amount))

#     'Done'




'''32. Longest Valid Parentheses'''
# def x():

#     from typing import Optional

#     # Input
#     # # Case 1
#     # s = "(()"
#     # # Output: 2

#     # Case 2
#     s = ")()())"
#     # Output: 4
    
#     # # Case 3
#     # s = ""
#     # # Output: 0
           
#     # # Custom Case
#     # s = "()"
#     # # Output: 0

#     '''
#     My Approach (Stack)

#         Intuition:
            
#             Based on what I learnt in the '20. Valid Parentheses' past leetcode challenge, I will try to modify it 
#             to make it work for this use case.
#     '''

#     def longestValidParentheses(s: str) -> int:

#         # Define the max string length holder
#         max_len = 0

#         # Handle Corner case: Empty string
#         if not s:
#             return max_len
        
#         # Initialize the variables to work with
#         stack = list(s)     # Generate a stack with the full input
#         temp = []           # Create a temp holder to check parentheses validity
#         temp_count = 0      # Create a temporary count to keep record of the running longest valid parentheses before uptading max_len

#         # Go thru the string character by character from right to left
#         while stack:

#             popped = stack.pop(-1)

#             # If the last popped char is a closing one, store it in the temp holder
#             if stack and popped == ')':
#                 temp.insert(0,popped)
            
#             else:
#                 # If the last stored char doens't match with the recently popped, means not a valid parentheses and the subtring to the right won't count for the next valid parenthesis, so update the running count and max count and reset the running count
#                 if not stack or not temp or popped == ')':
#                     max_len = max(max_len, temp_count)  # Update the max count to hold the max between the current max and the running max
#                     temp_count = 0                      # Reset the running count to start fresh a new count for the remaining string
#                     temp.clear()                        # Clear up the temp holder to star anew the running count            
                
#                 # Otherwise is a valid match
#                 else:
#                     temp_count += 2     # Add 2 to the temporary count, since '()' counts for 2
#                     temp = temp[1:]     # Take out the valid closing char from the temp holder   

#         # Return 'max_len'
#         return max_len

#     # Testing
#     print(longestValidParentheses(s=s))

#     '''
#     Note:     
#         This approach only solve 54% of test cases and has some issues managing the stack, opening parentheses and the frequency of the temp holders resetting.
#         Apparently there is a better way to manage this by tracking the string indices based on the same stack idea.
                
#     '''



#     '''
#     Optimized Approach (Stack)

#         Explanation of the new approach:

#             1. Using a stack of indices: We store the index of the last unmatched closing parenthesis ')' or the index of an unmatched opening parenthesis '('. This helps us calculate the length of valid substrings efficiently.

#             2. Initializing with -1: This handles the edge case where the string starts with a valid sequence. By initializing the stack with -1, we can easily calculate the length of the first valid substring.

#             3. Tracking indices, not characters: Instead of managing the characters directly, we work with the indices of parentheses. 
#                 This allows us to calculate the lengths of valid substrings by subtracting the index of the last unmatched parenthesis from the current index.

#             4. Calculating valid substring lengths: Each time a valid pair of parentheses is found (when the stack is not empty after popping), the length of the valid substring is i - stack[-1], 
#                 where i is the current index and stack[-1] is the index of the last unmatched parenthesis.

#             5. Edge cases: When encountering an unmatched closing parenthesis, we push its index onto the stack to handle any future valid subsequences that might occur after it.
#     '''

#     def longestValidParentheses(s: str) -> int:

#         # Initialize a stack with -1 to handle edge cases
#         stack = [-1]
#         max_len = 0

#         # Traverse the string
#         for i, char in enumerate(s):

#             if char == '(':
#                 # Push the index of the opening parenthesis
#                 stack.append(i)

#             else:
#                 # Pop the stack for a closing parenthesis
#                 stack.pop()

#                 if not stack:
#                     # If the stack is empty, push the current index
#                     stack.append(i)

#                 else:
#                     # Calculate the length of the valid substring
#                     max_len = max(max_len, i - stack[-1])
        
#         return max_len

#     # Testing
#     print(longestValidParentheses(s=s))

'''45. Jump Game II'''
# def x():

#     from typing import Optional

#     # Input
#     # Case 1
#     nums = [2,3,1,1,4]
#     # Output: 2 

#     # Case 2
#     nums = [2,3,0,1,4]
#     # Output: 2
    
#     # Custom Case
#     nums = [1,2,1,1,1]
#     # Output: 1

#     '''
#     My Approach - (Greedy)

#         Intuition:
            
#             - Initialize a jumps count in 1. (Because is a given that at least one jump must be done in any other than the corner case)
#             - In a while loop evaluate each option given by the next nums[i] element and pick the biggest.
#             - Return the number of jumps counted.
#     '''

#     def jump(nums: list[int]) -> int:

#         # Handle corner case: one-element input
#         if len(nums) == 1:
#             return 0
        
#         # Initilize a jump counter and a position holder
#         jumps = 1
#         pos = 0

#         # Traverse through the input elements
#         while pos < len(nums) - 1:
            
#             # Collect the possible next step with their respective positions
#             elems = []
#             for idx, num in enumerate(nums[ pos+1 : pos+nums[pos]+1 ], start=pos+1):
#                 elems.append((idx, num))

#             # Check if the last position is among the possibilities
#             pos_possibilities = [i[0] for i in elems]

#             if len(nums)-1 in pos_possibilities:
#                 return jumps

#             # Go to the position of the biggest possible jump / If the next possible have the same jump capacity get the farther
#             pos = max(elems, key=lambda x: x[1])[0]

#             # Add one jump to the counter
#             jumps += 1

#         # Return the jumps counter
#         return jumps


#     # Testing
#     # print(jump(nums = nums))

#     '''Note: This approach gets to solve 27% of test cases but the issue comes when two possible next steps have the same jump capacity, there, my logic breaks'''


#     '''
#     Greedy Approach Refined

#         Explanation
            
#             - Variables:

#                 *jumps: This keeps track of how many jumps you make.
#                 *farthest: The farthest index you can reach from the current position.
#                 current_end: The boundary of the current jump, i.e., how far you can go with the current number of jumps.
            
#             - Logic:

#                 * You iterate through the list and update farthest to track the farthest position you can reach from any position within the current range.
#                 * Whenever you reach current_end, it means you must make a jump, so you increase the jumps counter and set current_end to farthest.
#                 * You stop if current_end reaches or exceeds the last index because you know you can jump directly to the end.
#     '''  

#     def jump(nums: list[int]) -> bool:

#         # Handle corner case: single element input
#         if len(nums) == 1:
#             return 0
        
#         # Initialize variables
#         jumps = 0  # Number of jumps
#         farthest = 0  # The farthest point that can be reached
#         current_end = 0  # The farthest point within the current jump range
        
#         # Traverse the array, but we don't need to check the last element
#         for i in range(len(nums) - 1):

#             # Update the farthest point that can be reached
#             farthest = max(farthest, i + nums[i])
            
#             # If we reach the end of the current jump's range, we must make a new jump
#             if i == current_end:

#                 jumps += 1
#                 current_end = farthest  # Update the range of the next jump
                
#                 # If the farthest we can reach is the last index or beyond, we can stop
#                 if current_end >= len(nums) - 1:
#                     break
        
#         return jumps

        
#     # Testing
#     print(jump(nums = nums))

'''64. Minimum Path Sum'''
# def x():

#     from typing import Optional

#     # Input
#     # Case 1
#     grid = [[1,3,1],[1,5,1],[4,2,1]]
#     # Output: 7

#     # Case 2
#     grid = [[1,2,3],[4,5,6]]
#     # Output: 12

#     '''
#     My Approach (Dynamic Programming)

#         Intuition:
            
#             - Initialize a 'result' grid of dimentions m+1, n+1 and initilize the first row and the first column of each rown in 201*.
#             - Start traversing 'result' from result[1][1] and the operation will be to assign that cell: max( grid[i-1][j-1]+result[i][j-1], grid[i][j]+result[i-1][j1] )
#                 Except for the result[1][1], which will be assigned result[0][0]
#             - Return result[-1][-1] which will contain the shortest path to get there.

#             *201 is assigned as the value above the contrain '0 <= grid[i][j] <= 200', to say 'don't select this one'
#     '''

#     def minPathSum(grid: list[list[int]]) -> int:

#         # Capture the input dimentions
#         m, n = len(grid), len(grid[0])

#         # Handle Corner case: 1x1 input
#         if m == n == 1:
#             return grid[0][0]
        
#         # Handle Corner case: 1xn input
#         if m == 1:
#             return sum(grid[0])
        
#         # Handle Corner case: mx1 input
#         if m == 1:
#             return sum([x[0] for x in grid])
        

#         # Initialize the result grid holder
#         result = [[0]*(n+1) for _ in range(m+1)]

#         # Set the first row to 201
#         for col in range(len(result[0])):
#             result[0][col] = 201
        
#         # Set the first column to 201
#         for row in range(len(result)):
#             result[row][0] = 201

#         # Traverse the 'result' grid starting from [1][1]
#         for i in range(1, len(result)):
#             for j in range(1, len(result[0])):

#                 if j == i == 1:
#                     result[i][j] = grid[i-1][j-1]

#                 else:
#                     result[i][j] = min( grid[i-1][j-1]+result[i][j-1], grid[i-1][j-1]+result[i-1][j] )

#         # Return the last cell which is the one containing the shortest path
#         return result[-1][-1]

#     # Testing
#     print(minPathSum(grid=grid))

#     '''Note: My approach solved 96% of testcases'''


#     '''
#     My Approach (Dynamic Programming) - Correction
           
#         The initial idea was not wrong, but initializing with extra row and column could lead to miscalculations. 
#         So the 'result' holder will initialize only with the size of the grid and operation will be:
#             - For the first row: only summing up itself with the cell at the left.
#             - For the first column: only summing up itself with the cell at the top.
#             - For the rest of the result grid: Sum as planned.
#     '''


#     def minPathSum(grid: list[list[int]]) -> int:

#         # Capture the input dimentions
#         m, n = len(grid), len(grid[0])

#         # Handle Corner case: 1x1 input
#         if m == n == 1:
#             return grid[0][0]


#         # Initialize the result grid holder
#         result = [[0]*(n) for _ in range(m)]

#         # Initilize the top left cell of the result grid
#         result[0][0] = grid[0][0]


#         # Traverse the first row
#         for col in range(1, len(result[0])):
#             result[0][col] = result[0][col-1] + grid[0][col]
        
#         # Traverse the first column
#         for row in range(1, len(result)):
#             result[row][0] = result[row-1][0] + grid[row][0]

#         # Traverse the rest of 'result' grid starting from [1][1]
#         for i in range(1, len(result)):
#             for j in range(1, len(result[0])):
#                 result[i][j] = min( grid[i][j]+result[i][j-1], grid[i][j]+result[i-1][j] )

#         # Return the last cell which is the one containing the shortest path
#         return result[-1][-1]

#     # Testing
#     print(minPathSum(grid=grid))

#     '''Notes: Done!'''







