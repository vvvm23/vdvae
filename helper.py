info = lambda s: print(f"\33[92m> {s}\33[0m")
error = lambda s: print(f"\33[31m! {s}\33[0m")
debug = lambda s: print(f"\33[93m? {s}\33[0m") if DEBUG else None
warning = lambda s: print(f"\33[94m$ {s}\33[0m")
