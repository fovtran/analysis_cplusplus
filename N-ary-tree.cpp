// C++ implementation of the above approach
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define pb push_back
#define N 100005

// Keeping the values array indexed by 1.
int arr[8] = { 0, 1, 2, 2, 1, 4, 3, 6 };
vector<int> tree[N];

int idx, tin[N], tout[N], converted[N];

// Function to perform DFS in the tree
void dfs(ll node, ll parent)
{
    ++idx;
    converted[idx] = node;

    // To store starting range of a node
    tin[node] = idx;
    for (auto i : tree[node]) {
        if (i != parent)
            dfs(i, node);
    }

    // To store ending range of a node
    tout[node] = idx;
}

// Segment tree
ll t[N * 4];

// Build using the converted array indexes.
// Here a simple n-ary tree is converted
// into a segment tree.

// Now O(NlogN) range updates and queries
// can be performed.
void build(ll node, ll start, ll end)
{

    if (start == end)
        t[node] = arr[converted[start]];
    else {
        ll mid = (start + end) >> 1;
        build(2 * node, start, mid);
        build(2 * node + 1, mid + 1, end);

        t[node] = t[2 * node] + t[2 * node + 1];
    }
}

// Function to perform update operation
// on the tree
void update(ll node, ll start, ll end,
            ll lf, ll rg, ll c)
{
    if (start > end or start > rg or end < lf)
        return;

    if (start == end) {
        t[node] = c;
    }
    else {

        ll mid = (start + end) >> 1;
        update(2 * node, start, mid, lf, rg, c);
        update(2 * node + 1, mid + 1, end, lf, rg, c);

        t[node] = t[2 * node] + t[2 * node + 1];
    }
}

// Function to find the sum at every node
ll query(ll node, ll start, ll end, ll lf, ll rg)
{
    if (start > rg or end < lf)
        return 0;

    if (lf <= start and end <= rg) {
        return t[node];
    }
    else {
        ll ans = 0;
        ll mid = (start + end) >> 1;
        ans += query(2 * node, start, mid, lf, rg);

        ans += query(2 * node + 1, mid + 1,
                     end, lf, rg);

        return ans;
    }
}

// Function to print the tree
void printTree(int q, int node, int n)
{
    while (q--) {
        // Calculating range of node in segment tree
        ll lf = tin[node];
        ll rg = tout[node];
        ll res = query(1, 1, n, lf, rg);
        cout << "sum at node " << node
             << ": " << res << endl;
        node++;
    }
}

// Driver code
int main()
{
    int n = 7;
    int q = 7;

    // Creating the tree.
    tree[1].pb(2);
    tree[1].pb(3);
    tree[1].pb(4);
    tree[3].pb(5);
    tree[3].pb(6);
    tree[3].pb(7);

    // DFS to get converted array.
    idx = 0;
    dfs(1, -1);

    // Build segment tree with converted array.
    build(1, 1, n);

    printTree(7, 1, 7);

    // Updating the value at node 3
    int node = 3;
    ll lf = tin[node];
    ll rg = tout[node];
    ll value = 4;

    update(1, 1, n, lf, rg, value);

    cout << "After Update" << endl;
    printTree(7, 1, 7);

    return 0;
}
