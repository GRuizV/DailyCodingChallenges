'''
CHALLENGES INDEX

3. Longest Substring Without Repeating Characters (SW)
76. Minimum Window Substring (Hash Table) (SW)
239. Sliding Window Maximum (Array) (SW)
395. Longest Substring with At Least K Repeating Characters (SW) (RC) (DQ)

438. Find All Anagrams in a String (Hash-Table) (SW)
AgileEngine: Minimal Balls Move (SW)
567. Permutation in String (SW) (Hash Table)
209. Minimum Size Subarray Sum (SW) (Array)



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

(8)
'''


'3. Longest Substring Without Repeating Characters'
# def x():

#     from typing import Optional

#     # Input
#     # Case 1
#     s = "abcabcbb"
#     # Output: 3

#     # Case 2
#     s = "bbbbb"
#     # Output: 1

#     # Case 3
#     s = "pwwkew"
#     # Output: 3

#     '''
#     My Approach (Sliding Window)

#         Intuition:
            
#             - Handle corner case: string with no repeated chars, return len(s)

#             - Initialize an empty string holder 'window' to use it to process the input.

#             - Initialize a s_len integer holder in 0 to keep track of the length of the substrings processed

#             - In a for loop (for char in s):
                
#                 + If the current char is already in 'window' enter in a while-True loop:
                    
#                     * Redefine the 'window' as 'window[1:]'
#                     * If the current char is no longer in the window, end the while loop.

#                 + Update the s_len holder
            
#             - Return s_len
#     '''

#     def lengthOfLongestSubstring(s: str) -> int:

#         # Create a set from the input
#         s_set = set(s)

#         # Handle Corner case: string with no repeated chars, return len(s)
#         if len(s_set) == len(s):
#             return len(s)
        

#         #  Initialize an empty string holder
#         window = ''

#         # Initialize a s_len integer holder 
#         s_len = 0

#         # Process the input
#         for char in s:

#             if char in window:
                
#                 while True:
                    
#                     window = window[1:]

#                     if char not in window:
#                         break
            
#             window += char
#             s_len = max(s_len, len(window))


#         # Return the integer holder
#         return s_len

#     # Testing
#     print(lengthOfLongestSubstring(s=s))

#     '''Note: Done'''

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





'''438. Find All Anagrams in a String'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     s = "cbaebabacd"
#     p = "abc"
#     # Output: [0,6]

#     # Case 2
#     s = "abab"
#     p = "ab"
#     # Output: [0,1,2]

#     '''
#     My Approach (Brute forcing)

#         Intuition:
            
#             - Generate all possible permutations of characters in 'p' stored in a list holder.
#             - Create a 'ref' Hash-Table, with the first letter of each combination as key and a list* as value
#                 * This list will hold all combinations starting with its respective key
#             - Create a 'result' list holder, that will contain the indexes where an anagram appears.
#             - Iterate from 0 to (len(s)-len(p)) in a sliding window to validate for each s[i:i+len(p)] if s[i] is in 'ref'
#                 + if it does, the if s[i:i+len(p)-1] is in ref{s[i]}:
#                         + if it does, add i to 'result'
#     '''

#     from itertools import permutations

#     def findAnagrams(s: str, p: str) -> list[int]:

#         # Generate all possible permutations of 'p'
#         perms = [''.join(perm) for perm in permutations(p)]
        
#         # Create a reference Hash-Table containing all possible anagrams of 'p'
#         ref = {}

#         # Populate 'ref'
#         for per in perms:

#             # The first character will be the key and if doesn't exist yet within 'ref', create it.
#             if per[0] not in ref:   
#                 ref[per[0]] = []
            
#             # Add the permutation to its repective list in 'ref'
#             ref[per[0]].append(per)


#         # Create a result holder containing the occurrences indexes
#         result = []

#         # Traverse 's'
#         for i in range(len(s)-len(p)+1):
            
#             # If the initial character is a key in 'ref'
#             if s[i] in ref:
                
#                 elem = s[i:i+len(p)]

#                 # if the current slice (window) is in the key's list within 'ref'
#                 if elem in ref[s[i]]:

#                     # Add the index to result
#                     result.append(i)

#         # Return result
#         return result

#     # Testing
#     print(findAnagrams(s=s, p=p))

#     '''Note: My solution only met 31% of testcases'''




#     '''
#     Sliding Window Solution:

#         Explanation:
            
#             1. Use two frequency counters (dictionaries) for:

#                 * p_count: to store the frequency of characters in p.
#                 * window_count: to store the frequency of characters in the current window of s.
            
#             2. Slide a window of length len(p) across s. For each position:

#                 * Add the new character at the end of the window.
#                 * Remove the character that is no longer in the window.
#                 * Check if the current window has the same character frequencies as p_count.
           
#             3. If the frequency counts match, then the current window is an anagram of p, and you add the start index of the window to result.
#     '''

#     from collections import Counter

#     def findAnagrams(s: str, p: str) -> list[int]:
       
#         # Initialize result list
#         result = []

#         # Frequency counter for characters in p
#         p_count = Counter(p)

#         # Frequency counter for the current window in s
#         window_count = Counter()

#         # Length of the target anagram (p)
#         k = len(p)
        
#         # Slide over the string s
#         for i in range(len(s)):

#             # Add the current character to the window counter
#             window_count[s[i]] += 1

#             # Remove the leftmost character from the window counter if window is larger than k
#             if i >= k:

#                 # Leftmost character to remove
#                 left_char = s[i - k]

#                 if window_count[left_char] == 1:
#                     del window_count[left_char]
                
#                 else:
#                     window_count[left_char] -= 1

#             # Compare window counter with p counter
#             if window_count == p_count:
                
#                 # If they are equal, add the starting index of the window to the result
#                 result.append(i - k + 1)

#         return result
    
#     '''Note: Done'''

'AgileEngine: Minimal Balls Move'
# def min_moves_balls(buckets: str) -> int:

#     # Step 1: Count number of balls ('B')
#     ball_count = buckets.count('B')
    
#     # Corner case: No balls
#     if ball_count == 0:
#         return 0  # No balls, no moves needed
    
#     # Corner case: No enought spaces
#     if ball_count > buckets.count('.'):
#         return -1  # It's impossible to arrange with valid spacing
    
#     # Step 2: Initialize sliding window
#     start_index = 0
#     end_index = 2 * ball_count - 2  # Window size (the '-2' actually is just '-1' but since it'll be an index)
#     min_shifts = ball_count  # Start with max possible shifts
    
#     # Step 3: Slide the window across the string
#     while end_index < len(buckets):
        
#         ball_correct_pos = 0
#         window = buckets[start_index:end_index+1]   #This line is just to follow through the solution to understand it
        
#         # Count balls in correct positions (with gaps of 2 between them)
#         for i in range(start_index, end_index + 1, 2):
#             if buckets[i] == 'B':
#                 ball_correct_pos += 1
        
#         # Calculate the number of shifts needed for this window
#         shifts = ball_count - ball_correct_pos
#         min_shifts = min(min_shifts, shifts)
        
#         # Move the window
#         start_index += 1
#         end_index += 1
    
#     return min_shifts

'''567. Permutation in String'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     s1 = "ab"
#     s2 = "eidbaooo"
#     # Output: True

#     # Case 2
#     s1 = "ab"
#     s2 = "eidboaoo"
#     # Output: False
    
#     # Custome Case
#     s1 = "adc"
#     s2 = "dcda"
#     # Output: False

#     '''
#     My Approach (Sliding Window) (Hash Table)

#         Intuition:
            
#             - Handle corner case: if len(s1) > len(s2): False
#             - Create a counter of 's1'.
#             - Iterate through 's2' in a window of len(s1):
#                 + For each loop, create a counter with de window.
#                 + If this window counter is the same of the s1 counter, return True.
#             - If the code gets to finish the loop it means it didn't find a permutation of s1, return False.
#     '''
#     from collections import Counter

#     def checkInclusion(s1: str, s2: str) -> bool:

#         # Get the first string length
#         k = len(s1)

#         # Handle Corner case: first string bigger than the second
#         if k > len(s2):
#             return False
        
#         # Create a 's1' counter
#         s1_counter = Counter(s1)

#         # Traverse 's2'
#         for i in range(len(s2)-k+1):

#             # Create the temporary counter
#             temp_counter = Counter(s2[i:i+k])

#             # Check if the two counters coincide
#             if temp_counter == s1_counter:
#                 return True

#         # Return False if the loop ends
#         return False

#     # Testing
#     print(checkInclusion(s1=s1, s2=s2))

#     '''Note: This solution worked and beated submissions by 20% Runtime and 35% Memory'''

'''209. Minimum Size Subarray Sum'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     target = 7
#     nums = [2,3,1,2,4,3]
#     # Output: 2

#     # Case 2
#     target = 4
#     nums = [1,4,4]
#     # Output: 1

#     # Case 3
#     target = 11
#     nums = [1,1,1,1,1,1,1,1]
#     # Output: 0
    
#     # Custom Case
#     target = 213
#     nums = [12,28,83,4,25,26,25,2,25,25,25,12]
#     # Output: 8
    
#     '''
#     My Approach (Sliding Window)

#         Intuition:
            
#             - Handle corner case: if the input sum is less than 'target' return 0.
            
#             - initialize 'min_len' holder in float('inf').

#             - Initialize a 'window' holding wich will hold the resulting subarray to be evaluated.

#             - Initialize two indices 'l' and 'r' in 0 and '1', that will drive the respective ends of the 'window'.

#             - In a while loop (Until 'l' is less than len(nums)) process the input:
                
#                 + if sum(window[l:r]) >= target:
#                     min_len = min(min_len, len(window[l:r]))
#                     l += 1
#                     r = l
                
#                 + if sum(window[l:r]) < target:                    
#                     * r += 1
            
#             - retun 0 if min_len == float('inf') else retun min_len
#     '''

#     def minSubArrayLen(target: int, nums: list[int]) -> int:

#         # Handle Corner case: the input sum is less than 'target' return 0
#         if sum(nums) < target:
#             return 0
  
#         # initialize 'min_len' holder in float('inf')
#         min_len = float('inf')

#         # Initialize a 'window' holding wich will hold the resulting subarray to be evaluated.
#         window = []

#         # Initialize two indices 'l' and 'r' in 0
#         l = r = 0

#         # Process 'nums'
#         while l < len(nums):

#             window = nums[l:r]

#             if r > len(nums):
#                 l += 1
#                 r = l
            
#             else: 
#                 if sum(window) >= target:

#                     min_len = min(min_len, len(window))
#                     l += 1
#                     r = l
                
#                 if sum(window) < target:                    
#                     r += 1
        
#         # Return the result
#         return 0 if min_len == float('inf') else min_len


#     # Testing
#     print(minSubArrayLen(target=target, nums=nums))

#     '''
#     Notes: 
#         - In my first attempt, my solution met 85% of testcases, which is great for me given that I'm knowlegdeable regaring Sliding Window technique.
#     '''
    

    
#     '''
#     Optimized Sliding Window Approach
#     '''

#     def minSubArrayLen(target: int, nums: list[int]) -> int:
#         l = 0
#         window_sum = 0
#         min_len = float('inf')

#         for r in range(len(nums)):
#             window_sum += nums[r]
            
#             # Shrink the window as much as possible while still meeting the target
#             while window_sum >= target:
#                 min_len = min(min_len, r - l + 1)
#                 window_sum -= nums[l]
#                 l += 1

#         return 0 if min_len == float('inf') else min_len











