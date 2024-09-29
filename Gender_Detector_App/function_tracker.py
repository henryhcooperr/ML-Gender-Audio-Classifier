# Define a global counter dictionary to store the count for each function
function_counter = {}
on = False
# Decorator function to track the number of times a function is called
def count_function_calls(func):
    def wrapper(*args, **kwargs):
        # Increment the counter for the function
        function_name = f"{func.__module__}.{func.__name__}"
        if function_name not in function_counter:
            function_counter[function_name] = 1
        else:
            function_counter[function_name] += 1
        
        # Print the count and function name before executing the function
        if on == True:
            print(f"{function_counter[function_name]}: ran {function_name}(): ...")

        # Call the original function
        return func(*args, **kwargs)
    return wrapper