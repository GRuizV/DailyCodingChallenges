'''
CHALLENGES INDEX

5. Longest Palindromic Substring (DP) (TP)
22. Generate Parentheses (DP) (BT)


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












