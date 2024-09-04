'''
CHALLENGES INDEX

3. Longest Substring Without Repeating Characters (Hash Table) (SW)
76. Minimum Window Substring (Hash Table) (SW)
239. Sliding Window Maximum (Array) (SW)


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
*Arrays, Hash Tables & Matrices
*Sorting
*Heaps, Stacks & Queues
*Graphs, Trees & Binary Trees

(XX)
'''


'3. Longest Substring Without Repeating Characters'
# def x():

#     s = "abcabcbb"


#     # My solution
#     substrings = []

#     i = 0

#     while i < len(s):

#         sub = str()

#         for char in s[i:]:

#             if char in sub:
#                 substrings.append(sub)
#                 break

#             sub += char
        
#         if sub not in substrings:
#             substrings.append(sub)

#         i += 1

#     # print(substrings)

#     max_sub = max(substrings, key = len) if substrings else 0

#     # print(max_sub)

#     print(max_sub, len(max_sub))


#     # Another more efficient solution

#     def lengthOfLongestSubstring(s: str) -> int:
            
#             n = len(s)
#             maxLength = 0
#             charMap = {}
#             left = 0
            
#             for right in range(n):

#                 if s[right] not in charMap or charMap[s[right]] < left:
#                     charMap[s[right]] = right
#                     maxLength = max(maxLength, right - left + 1)

#                 else:
#                     left = charMap[s[right]] + 1
#                     charMap[s[right]] = right
            
#             return maxLength


#     lengthOfLongestSubstring(s)

'76. Minimum Window Substring'
# def x():

#     # Input
#     # Case 1
#     s, t = 'ADOBECODEBANC', 'ABC'
#     # Output: "BANC"

#     # Case 2
#     s, t = 'a', 'a'
#     # Output: "a"

#     # Case 3
#     s, t = 'a', 'aa'
#     # Output: "abbbbbcdd"

#     # Custom case
#     s, t = 'aaaaaaaaaaaabbbbbcdd', 'abcdd'
#     # Output: "abbbbbcdd"


#     'My approach'
#     def minWindow(s:str, t:str) -> str:

#         if len(t) > len(s):
#             return ''
        
#         if t == s:
#             return t
        

#         for i in range(len(t), len(s) + 1):

#             for j in range((len(s)-i) + 1):
                
#                 if all([char in s[j:j+i] for char in t]):
#                     return s[j:j+i]
                
#         return ''

#     'Notes: This solution works up to 57%'


#     'With an improvement'
#     def minWindow(s:str, t:str) -> str:

#         from collections import Counter

#         if len(t) > len(s):
#             return ''
        
#         if t == s:
#             return t
        
#         count_t = Counter(t).items()

#         for i in range(len(t), len(s) + 1):

#             for j in range((len(s)-i) + 1):
                
#                 subs = s[j:j+i]
#                 count_subs = Counter(subs)

#                 if all( (x[0] in count_subs.keys() and x[1] <= count_subs[x[0]]) for x in count_t):
#                     return s[j:j+i]
                
#         return ''

#     'Notes: This solution works up to 93% and hit the time limit'


#     'Another solution'
#     def minWindow(s, t):    

#         if not s or not t:
#             return ""


#         from collections import defaultdict

#         dictT = defaultdict(int)
#         for c in t:
#             dictT[c] += 1

#         required = len(dictT)
#         l, r = 0, 0
#         formed = 0

#         windowCounts = defaultdict(int)
#         ans = [-1, 0, 0]

#         while r < len(s):
#             c = s[r]
#             windowCounts[c] += 1

#             if c in dictT and windowCounts[c] == dictT[c]:
#                 formed += 1

#             while l <= r and formed == required:
#                 c = s[l]

#                 if ans[0] == -1 or r - l + 1 < ans[0]:
#                     ans[0] = r - l + 1
#                     ans[1] = l
#                     ans[2] = r

#                 windowCounts[c] -= 1
#                 if c in dictT and windowCounts[c] < dictT[c]:
#                     formed -= 1

#                 l += 1

#             r += 1

#         return "" if ans[0] == -1 else s[ans[1]:ans[2] + 1]
            
#     # Testing
#     print(minWindow(s,t))

'''239. Sliding Window Maximum'''
# def x():

#     # Input
#     # Case 1
#     nums = [1,3,-1,-3,5,3,6,7]
#     k = 3
#     # Output: [3,3,5,5,6,7]

#     # Case 2
#     nums = [1]
#     k = 1
#     # Output: [1]

#     # Cusom Case
#     nums = [1,3,-1,-3,5,3,6,7]
#     k = 3
#     # Output: [3,3,5,5,6,7]


#     'My approach'
#     def max_sliding_window(nums:list[int], k:int) -> list[int]:

#         if len(nums) == 1:
#             return nums
        
#         if k == len(nums):
#             return [max(nums)]


#         result = []

#         for i in range(len(nums)-k+1):
#             result.append(max(nums[i:i+k]))

#         return result

#     # Testing
#     print(max_sliding_window(nums=nums, k=k))

#     'Note: This approach cleared 73% of test cases, but breaks with large inputs'


#     'Monotonically Decreacing Queue'
#     def max_sliding_window(nums:list[int], k:int) -> list[int]:

#         import collections

#         output = []
#         deque = collections.deque() # nums
#         left = right = 0

#         while right < len(nums):

#             # Pop smaller values from de deque
#             while deque and nums[deque[-1]] < nums[right]:
#                 deque.pop()

#             deque.append(right)

#             # remove the left val from the window
#             if left > deque[0]:
#                 deque.popleft()

#             if (right+1) >= k:
#                 output.append(nums[deque[0]])
#                 left += 1
            
#             right += 1

#         return output

#     # Testing
#     print(max_sliding_window(nums=nums, k=k))

#     'done'











