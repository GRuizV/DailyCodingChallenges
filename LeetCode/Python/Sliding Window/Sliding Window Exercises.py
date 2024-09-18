'''
CHALLENGES INDEX

3. Longest Substring Without Repeating Characters (Hash Table) (SW)
76. Minimum Window Substring (Hash Table) (SW)
239. Sliding Window Maximum (Array) (SW)
395. Longest Substring with At Least K Repeating Characters (SW) (RC) (DQ)


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

(4)
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

'''395. Longest Substring with At Least K Repeating Characters'''
# def x():

#     # Input
#     # Case 1
#     s, k = "aaabb", 3
#     # Output: 3 / The longest substring is "aaa", as 'a' is repeated 3 times.

#     # Case 2
#     s, k = "ababbc", 2
#     # Output: 5 / The longest substring is "aaa", as 'a' is repeated 3 times.


#     '''
#     My approach

#         Intuition:
            
#             Brute forcing:

#                 - Import the Counter class from collections.
#                 - Initialize a max_len counter in 0 to hold the max len of a valid substring according to the requirements of k.
#                 - Starting from the len(s) down to k, check in a range, all the substrings of all those different sizes and
#                     with Counter's help check is the minimum freq is at least k,
#                         if it does: Refresh the max_len counter.
#                         if it doesn't: check the rests of the substrings.
#     '''

#     from collections import Counter

#     def longestSubstring(s: str, k: int) -> int:

#         # Initialize the max counter
#         max_len = 0

#         # Capture the len of s
#         l = len(s)

#         # Handle the corner case: len(s) < k
#         if l < k:
#             return max_len

#         # Check all possibles valid substrings
#         for i in range(k-1, l):

#             for j in range(l-i):

#                 # Create the possible valid substring
#                 substring = s[j:j+i+1]

#                 # Create a counter from the substring
#                 subs_counter = Counter(substring)

#                 # Capture the minimum freq of the caracters present
#                 subs_min_freq = min(subs_counter.values())

#                 # Update the counter only if the minimum is at least k in size
#                 max_len = len(substring) if subs_min_freq >= k else max_len


#         # Return what's un the max counter
#         return max_len

#     # Testing
#     print(longestSubstring(s=s, k=k))

#     'Note: This approach met the 87% of cases but with large input breaks. I will rethink the loop to make it go from the largest to the lowest limit, that should save some runtime.'


#     'My 2nd approach'
#     from collections import Counter

#     def longestSubstring(s: str, k: int) -> int:

#         # Capture the len of s
#         l = len(s)

#         # Handle the corner case: len(s) < k
#         if l < k:
#             return 0

#         # Check all possibles valid substrings
#         for i in range(l-1, k-2, -1):

#             if i != -1:

#                 for j in range(l-i):
                            
#                     # Create the possible valid substring
#                     substring = s[j:j+i+1]

#                     # Create a counter from the substring
#                     subs_counter = Counter(substring)

#                     # Capture the minimum freq of the caracters present
#                     subs_min_freq = min(subs_counter.values())

#                     # If the min freq found is at least k, that's the longest valid substring possible
#                     if subs_min_freq >= k:
#                         return len(substring)

#         # Return 0
#         return 0

#     # Testing
#     print(longestSubstring(s=s, k=k))

#     'Note: Unfortunately my second approach had the same performance.'


#     'Divide and Conquer approach'
#     from collections import Counter

#     def longestSubstring(s: str, k: int) -> int:

#         # Base case
#         if len(s) == 0 or len(s) < k:
#             return 0

#         # Count the frequency of eachcharacter in the string
#         counter = Counter(s)

#         # Iterate through the string and split at a character that doesn't meet the frequency requirement
#         for i, char in enumerate(s):

#             if counter[char] < k:

#                 # Split and recursively process the left and right substrings
#                 left_part = longestSubstring(s[:i], k)
#                 right_part = longestSubstring(s[i+1:], k)

#                 return max(left_part, right_part)

#         # If there's no splits, means that the entire substring is valid
#         return len(s)

#     'Done'










