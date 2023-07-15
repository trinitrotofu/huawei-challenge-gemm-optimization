// This is just for analyzing the data set
#include <bits/stdc++.h>

using namespace std;

#define int long long

bool cmp(pair<int, int> &p1, pair<int, int> &p2) {
    if (p1.second == p2.second) return p1.first > p2.first;
    return p1.second > p2.second;
}

int32_t main() {
//    freopen("gemm_inputs/0.txt", "r", stdin);

//    string s;
//    cin >> s;
//    cout << s << '\n';

//    freopen("outputs/out.txt", "w+", stdout);
//

    for (int i = 0; i <= 9; i++) {
        map<int, int> mp;
        int cnt = 0;
        string filename = "gemm_inputs/";
        filename += to_string(i);
        filename += ".txt";
        ifstream read(filename);
        string m, n, k;
        int biggest = 0;
        int cnt_small = 0;
        int sq = 0;
        int total = 0;
        while (getline(read, m, ',') && getline(read, n, ',') && getline(read, k, ',')) {
            cnt += 3;
//            mp[m]++, mp[n]++, mp[k]++;
            biggest = max(biggest, (long long) stoi(m) * stoi(n));
            biggest = max(biggest, (long long) stoi(m) * stoi(k));
            biggest = max(biggest, (long long) stoi(k) * stoi(n));
//            total += stoi(m) * stoi(k);
//            total += stoi(k) * stoi(n);
//            total += stoi(m) * stoi(n);
//            mp[stoi(m) * stoi(n)]++;
//            mp[stoi(n) * stoi(k)]++;
//            int total = stoi(m) * stoi(n) * stoi(k);
//            if (total < 80 * 80 * 80) cnt_small++;

//            if (m == n && n == k) {
//                sq++;
//                cout << m << " x " << m << '\n';
//            }
        }
//        vector<pair<int, int>> freq;
//        for (auto p: mp) freq.push_back({p.second, p.first});
//        sort(freq.begin(), freq.end(), greater<pair<int, int>>());
//        sort(freq.begin(), freq.end(), cmp);
//        cout << "File " << i << ": " << cnt / 3 << '\n';
//        cout << "File " << i << ": " << '\n';
//        cout << cnt1 << '\n';

//        cout << "Biggest: " << biggest << '\n';
//        cout << mp[to_string(biggest)] << '\n';

//            cout.precision(3);
//        for (int j = 0; j < 20; j++) {
//            cout << fixed << (long double) freq[j].first / cnt * 100 << "% of this file is " << freq[j].second << "'s" << '\n';
//        }
//        cout << biggest << '\n';
        cout << cnt << '\n';
//        cout << cnt_small << '\n';
//        cout << (long double) cnt_small / cnt << '\n';
    }
}

/*
2253756
8271135
2798511
1029159
1025547
3908421
3076236
3346998
4652490
5518902
 */
