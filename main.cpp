#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <stack>
#include <unordered_set>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <iomanip> // std::setprecision

int D = 0;
double* solution;

struct point {
	static double eps;
	double x = 0, y = 0, z = 0;
	point operator+(const point& rhs) const {
		return point{ x + rhs.x,y + rhs.y,z + rhs.z };
	}
	point operator*(const double& rhs) const {
		return point{ x * rhs,y * rhs,z * rhs };
	}
	point operator/(const double& rhs) const {
		return point{ x / rhs,y / rhs,z / rhs };
	}
	point operator-(const point& rhs) const {
		return point{ x - rhs.x,y - rhs.y,z - rhs.z };
	}
	bool operator==(const point& rhs) const {
		return (fabs(x - rhs.x) < eps && fabs(y - rhs.y) < eps && fabs(z - rhs.z) < eps);
	}
};
double point::eps = 1e-8;
point operator*(const double& weight, const point& p) {
	return point{ weight * p.x,weight * p.y,weight * p.z };
}

struct normal {
	double x = 0, y = 0, z = 0;
	double operator*(const normal& rhs) const {
		return x * rhs.x + y * rhs.y + z * rhs.z;
	}
	normal operator*(const double& rhs) const {
		return normal{ x * rhs,y * rhs,z * rhs };
	}
	void operator*=(const double& rhs) {
		x *= rhs;
		y *= rhs;
		z *= rhs;
	}
	normal operator+(const normal& rhs) const {
		return normal{ x + rhs.x,y + rhs.y,z + rhs.z };
	}
	normal operator/(const double& rhs) const {
		return normal{ x / rhs,y / rhs,z / rhs };
	}
	void operator/=(const double& rhs) {
		x /= rhs;
		y /= rhs;
		z /= rhs;
	}
	normal operator-(const normal& rhs) const {
		return normal{ x - rhs.x,y - rhs.y,z - rhs.z };
	}
	void operator+=(const normal& rhs) {
		x += rhs.x;
		y += rhs.y;
		z += rhs.z;
	}
};
normal operator*(const double& weight, const normal& v) {
	return normal{ weight * v.x,weight * v.y,weight * v.z };
}

int n; // 采样点个数
point* a; // 所有采样点的集合
point* sorted_a;
normal* va; // 所有采样点法向量的集合
normal* sorted_va;

struct polynomial {
	int degree; // 多项式次数
	double* coeffs; // 系数，共degree+1个
	polynomial(int input_degree) {
		degree = input_degree;
		coeffs = new double[degree + 1]();
	}
	~polynomial() {
		delete[] coeffs;
	}
	// 禁用拷贝构造函数和拷贝赋值运算符
	polynomial(const polynomial&) = delete;
	polynomial& operator=(const polynomial&) = delete;
	polynomial(polynomial&& other) noexcept { // 移动构造
		degree = other.degree;
		coeffs = other.coeffs;
		other.coeffs = nullptr;
	}
	polynomial& operator=(polynomial&& other) noexcept { // 移动赋值
		if (this == &other) return *this; // 防止自赋值
		delete[] coeffs;
		degree = other.degree;
		coeffs = other.coeffs;
		other.coeffs = nullptr;
		return *this;
	}
	double operator()(double x) const {
		double res = 0;
		for (int i = 0;i <= degree;i++) {
			res += coeffs[i] * pow(x, i);
		}
		return res;
	}
	polynomial operator*(const polynomial &rhs) const {
		polynomial res(degree + rhs.degree);
		res.coeffs = new double[res.degree + 1](); // 括号是默认初始化为零值
		for (int i = 0;i <= degree;i++) {
			for (int j = 0;j <= rhs.degree;j++) {
				int k = i + j;
				res.coeffs[k] += coeffs[i] * rhs.coeffs[j];
			}
		}
		return res;
	}
	void print() {
		for (int i = 0;i <= degree;i++) {
			printf("%.9lf x^%d", coeffs[i], i);
			if (i < degree) printf(" + ");
		}
		printf("\n");
	}
	void output(std::ofstream& file) {
		for (int i = 0;i <= degree;i++) {
			file << coeffs[i] << " ";
		}
	}
};

struct function {
	// 需要用到的函数都可以用分段多项式表示
	int num; // 多项式个数
	int degree; // 多项式最高次数
	polynomial* polys; // 多项式序列，数量为num
	double* break_points; // 分段点序列，数量为num+1，升序排列
	function(int input_num, int input_degree) {
		num = input_num;
		degree = input_degree;
		void* p = operator new[](num * sizeof(polynomial));
		polys = static_cast<polynomial*>(p);
		for (int i = 0;i < num;i++) {
			new(&polys[i]) polynomial(degree);
		}
		break_points = new double[num + 1]();
	}
	~function() {
		if (polys != nullptr) {
			for (int i = 0;i < num;i++) {
				polys[i].~polynomial();
			}
		}
		delete[] break_points;
	}
	// 禁用拷贝构造函数和拷贝赋值运算符
	function(const function&) = delete;
	function& operator=(const function&) = delete;
	function(function&& other) noexcept { // 移动构造
		num = other.num;
		degree = other.degree;
		polys = other.polys;
		break_points = other.break_points;
		other.polys = nullptr;
		other.break_points = nullptr;
	}
	function& operator=(function&& other) noexcept { // 移动赋值
		if (this == &other) return *this; // 防止自赋值
		num = other.num;
		degree = other.degree;
		for (int i = 0;i < num;i++) {
			polys[i] = std::move(other.polys[i]);
		}
		delete[] break_points;
		break_points = other.break_points;
		other.break_points = nullptr;
		return *this;
	}
	double operator()(double x) const {
		for (int i = 0;i < num;i++) {
			if (break_points[i] <= x && x <= break_points[i + 1]) {
				return polys[i](x);
			}
		}
		return 0;
	}
	void print() {
		for (int i = 0;i < num;i++) {
			printf("[%.9lf, %.9lf]: ", break_points[i], break_points[i + 1]);
			polys[i].print();
		}
	}
	void output(std::ofstream& file) {
		file << std::fixed << std::setprecision(9);
		for (int i = 0;i < num;i++) {
			file << break_points[i] << " " << break_points[i + 1] << " ";
			polys[i].output(file);
			file << std::endl;
		}
	}
};

function F(3, 2); // F是一元标准基函数

double integral(const polynomial& f, double l, double r) {
	// 对多项式f在区间[l,r]上积分
	if (fabs(l - r) < 1e-15) return 0;
	double res = 0;
	for (int i = 0;i <= f.degree;i++) {
		res += f.coeffs[i] * (pow(r, i + 1) - pow(l, i + 1)) / (i + 1);
	}
	return res;
}

double integral(const function& f) {
	// 对函数f在实数轴上积分
	double res = 0;
	for (int i = 0;i < f.num;i++) {
		res += integral(f.polys[i], f.break_points[i], f.break_points[i + 1]);
	}
	return res;
}

function differential(const function& f) {
	// 我们构造的F的分段点处不一定可导（按论文2的设置，只与自身卷了一次），但没关系，我们求导是为了做内积，有限点处不可导不影响积分结果
	function res(f.num, std::max(0, f.degree - 1));
	for (int i = 0;i <= res.num;i++) {
		res.break_points[i] = f.break_points[i];
	}
	for (int i = 0;i < f.num;i++) {
		polynomial& p = f.polys[i];
		polynomial& p2 = res.polys[i];
		if (p.degree > 0) {
			for (int j = 0;j <= p2.degree;j++) {
				p2.coeffs[j] = p.coeffs[j + 1] * (j + 1);
			}
		}
		else {
			p2.coeffs[0] = 0;
		}
	}
	return res;
}

double inner_product(const function& a, const function& b) {
	// 负责计算两个分段多项式的函数内积
	int idx1 = 0, idx2 = 0; // idx1为当前段对应的a的多项式编号，idx2为当前段对应的b的多项式编号
	double l = 0, r = 0;
	double res = 0;
	while (idx1 < a.num && idx2 < b.num) {
		int* idx_l = nullptr;
		int* idx_r = nullptr;
		if (a.break_points[idx1] > b.break_points[idx2]) {
			l = a.break_points[idx1];
			idx_l = &idx2;
		}
		else {
			l = b.break_points[idx2];
			idx_l = &idx1;
		}
		if (a.break_points[idx1 + 1] < b.break_points[idx2 + 1]) {
			r = a.break_points[idx1 + 1];
			idx_r = &idx1;
		}
		else {
			r = b.break_points[idx2 + 1];
			idx_r = &idx2;
		}
		if (l > r) {
			(*idx_l)++;
			continue;
		}
		res += integral(a.polys[idx1] * b.polys[idx2], l, r);
		(*idx_r)++;
	}
	return res;
}

function get_Fo(double center_x, double width) {
	// center_x 是中心点沿某个轴的坐标分量
	// Fo(x) = F((x - center_x) / width)
	function Fo(3, 2);
	for (int i = 0;i < 3;i++) {
		Fo.polys[i].coeffs[0] = F.polys[i].coeffs[0] - F.polys[i].coeffs[1] * center_x / width + F.polys[i].coeffs[2] * center_x * center_x / (width * width);
		Fo.polys[i].coeffs[1] = F.polys[i].coeffs[1] / width - F.polys[i].coeffs[2] * 2 * center_x / (width * width);
		Fo.polys[i].coeffs[2] = F.polys[i].coeffs[2] / (width * width);
	}
	for (int i = 0;i < 4;i++) Fo.break_points[i] = F.break_points[i] * width + center_x;
	return Fo;
}

struct x_bits {
	unsigned int value = 0; // 32位二进制串value，表示x每一次的走向，x泛指任意一个坐标分量
	int d = 0; // value从低到高的d位是有效位
	void init(unsigned int input_value, int input_d) {
		value = input_value;
		d = input_d;
	}
	double get_center_x(double width) {
		double center_x = 0; // 因为我们总是把中心放在(0,0,0)，所以一开始坐标分量一定是0
		for (int i = 0; i < d;i++) {
			if (value & (1 << (d - 1 - i))) {
				center_x += width / 4;
			}
			else {
				center_x -= width / 4;
			}
			width /= 2;
		}
		return center_x;
	}
};

struct xyz_key {
	static unsigned int branch_mask;
	static unsigned int parent_mask;
	unsigned int value = 0; // 32位二进制串value
	int d = 0; // value低3d位是有效位，从高到低写出来是x1y1z1 x2y2z2 ... xdydzd
	void init(unsigned int input_value, int input_d) {
		value = input_value;
		d = input_d;
	}
	void compute(int idx_p, int D, double width) {
		// 计算value,d
		d = D;
		point center{ 0,0,0 };
		const point& p = a[idx_p];
		value = 0;
		for (int i = 0;i < D;i++) {
			if (p.x > center.x) {
				value |= 4 << 3 * (D - 1 - i);
				center.x += width / 4;
			}
			else center.x -= width / 4;
			if (p.y > center.y) {
				value |= 2 << 3 * (D - 1 - i);
				center.y += width / 4;
			}
			else center.y -= width / 4;
			if (p.z > center.z) {
				value |= 1 << 3 * (D - 1 - i);
				center.z += width / 4;
			}
			else center.z -= width / 4;
			width /= 2;
		}
	}
	x_bits extract(int idx) {
		// idx: 0表示x 1表示y 2表示z
		int c = 0;
		if (idx == 0) c = 4;
		else if (idx == 1) c = 2;
		else c = 1;
		unsigned int res = 0;
		for (int k = 0;k < d;k++) {
			res <<= 1;
			if (value & (c << 3 * (d - 1 - k))) res |= 1;
		}
		return x_bits{ res, d };
	}
	unsigned int get_parent_key() {
		return ((value & parent_mask) >> 3);
	}
	unsigned int get_branch_key() {
		return (value & branch_mask);
	}
	bool operator<(const xyz_key& rhs) const {
		return value < rhs.value;
	}
	bool operator==(const xyz_key& rhs) const {
		return (value == rhs.value) && (d == rhs.d);
	}
	void print() {
		for (int i = 0;i < 3 * d;i++) {
			std::cout << ((value & (1 << (3 * d - 1 - i))) ? 1 : 0);
		}
		std::cout << std::endl;
	}
};
unsigned int xyz_key::branch_mask = ((1 << 3) - 1);
unsigned int xyz_key::parent_mask = ~xyz_key::branch_mask;

struct code {
	xyz_key key;
	int idx;
	bool operator<(const code& rhs) const {
		if (key == rhs.key) {
			return idx < rhs.idx;
		}
		else {
			return key < rhs.key;
		}
	}
};
code* codes;


// 三进制编码，方便枚举节点邻居
constexpr int fromTernaryChar(char c) {
	return c - '0';
}

constexpr int fromTernary(const char* ternary) {
	int result = 0;
	for (; *ternary != '\0'; ++ternary) {
		result = result * 3 + fromTernaryChar(*ternary);
	}
	return result;
}


struct node;

// 节点上的操作
struct operation {
	virtual void operator()(node* a, node* b) = 0; // 纯虚函数
};

// 八叉树节点
struct node {
	double width; // 对应立方体的宽度
	point center; // 对应立方体的中心点
	normal v; // 该节点的v（向量）
	node* parent = nullptr; // 该节点的父指针
	bool has_children = false; // 该节点是否有儿子节点
	node* children[8] = { nullptr }; // 该节点的儿子节点指针
	node* neighbors[27] = { nullptr }; // 该节点的同深度的邻居节点数组
	xyz_key key; // 节点的shuffled xyz key
	int idx_p = -1; // 该节点所含有的编号最小的采样点的编号（编号连续）
	int cnt_p = 0; // 该节点所含有的采样点的个数
	int idx_o = -1; // 该节点所含有的最底层节点在同层节点数组中的起始编号（编号连续）
	int cnt_o = 0; // 该节点所含有的最底层节点的个数
	int depth = 0; // 节点深度
	double b = 0; // 该节点的散度值
	int idx_node = 0; // 该节点的编号
	double sampling_density = 0; // 对应立方体的采样密度
	void init_children() {
		has_children = true;
		point center_child;
		for (int i = 0;i < 2;i++) {
			if (i == 0) center_child.x = center.x - width / 4;
			else center_child.x = center.x + width / 4;
			for (int j = 0;j < 2;j++) {
				if (j == 0) center_child.y = center.y - width / 4;
				else center_child.y = center.y + width / 4;
				for (int k = 0;k < 2;k++) {
					if (k == 0) center_child.z = center.z - width / 4;
					else center_child.z = center.z + width / 4;
					int idx = 4 * i + 2 * j + 1 * k;
					children[idx] = new node();
					children[idx]->center = center_child;
					children[idx]->width = width / 2;
					children[idx]->depth = depth + 1;
					children[idx]->parent = this;
					children[idx]->idx_node = -1; // 用于evaluate的时候跳过细分产生的节点，因为解向量x在这些节点上没有值
				}
			}
		}
	}
	point get_vertex(int idx) {
		point vertex = center;
		if (idx & 4) vertex.x += width / 2;
		else vertex.x -= width / 2;
		if (idx & 2) vertex.y += width / 2;
		else vertex.y -= width / 2;
		if (idx & 1) vertex.z += width / 2;
		else vertex.z -= width / 2;
		return vertex;
	}
	void compute_center() {
		// 要求已知key, width
		double root_width = width * (1 << depth);
		x_bits xbits;
		xbits = key.extract(0);
		center.x = xbits.get_center_x(root_width);
		xbits = key.extract(1);
		center.y = xbits.get_center_x(root_width);
		xbits = key.extract(2);
		center.z = xbits.get_center_x(root_width);
	}
	void naive_compute_neighbors() {
		// 要求已知parent的neighbors
		static int naive_table[27][3] = {
			{fromTernary("000"), 0b000, 0b111},
			{fromTernary("001"), 0b000, 0b110},
			{fromTernary("002"), 0b001, 0b110},
			{fromTernary("010"), 0b000, 0b101},
			{fromTernary("011"), 0b000, 0b100},
			{fromTernary("012"), 0b001, 0b100},
			{fromTernary("020"), 0b010, 0b101},
			{fromTernary("021"), 0b010, 0b100},
			{fromTernary("022"), 0b011, 0b100},
			{fromTernary("100"), 0b000, 0b011},
			{fromTernary("101"), 0b000, 0b010},
			{fromTernary("102"), 0b001, 0b010},
			{fromTernary("110"), 0b000, 0b001},
			{fromTernary("111"), 0b000, 0b000},
			{fromTernary("112"), 0b001, 0b000},
			{fromTernary("120"), 0b010, 0b001},
			{fromTernary("121"), 0b010, 0b000},
			{fromTernary("122"), 0b011, 0b000},
			{fromTernary("200"), 0b100, 0b011},
			{fromTernary("201"), 0b100, 0b010},
			{fromTernary("202"), 0b101, 0b010},
			{fromTernary("210"), 0b100, 0b001},
			{fromTernary("211"), 0b100, 0b000},
			{fromTernary("212"), 0b101, 0b000},
			{fromTernary("220"), 0b110, 0b001},
			{fromTernary("221"), 0b110, 0b000},
			{fromTernary("222"), 0b111, 0b000},
		};
		neighbors[fromTernary("111")] = this;
		if (parent == nullptr) return;
		for (int i = 0;i < 27;i++) {
			if (parent->neighbors[i] == nullptr || !parent->neighbors[i]->has_children) {
				continue;
			}
			for (int j = 0;j < 8;j++) {
				node& candidate = *parent->neighbors[i]->children[j];
				for (int dir = 0;dir < 27;dir++) {
					// 测试candidate是否为当前节点dir方向上的邻居
					if (get_vertex(naive_table[dir][1]) == candidate.get_vertex(naive_table[dir][2])) {
						neighbors[dir] = &candidate;
						break;
					}
				}
			}
		}
	}
	void compute_neighbors_by_LUT() {
		// 要求已知parent的neighbors

	}
	void process(node* cur, operation* op) {
		// 遍历基函数与当前节点基函数有重合的所有节点，并执行相应操作
		(*op)(this, cur);
		if (cur->has_children) {
			for (int i = 0;i < 8;i++) {
				node* o = cur->children[i];
				if (o->idx_node == -1) continue;
				double dis = 1.5 * (o->width + width);
				if (fabs(o->center.x - center.x) < dis && fabs(o->center.y - center.y) < dis && fabs(o->center.z - center.z) < dis) {
					process(o, op);
				}
			}
		}
	}
};

// 假设D < 10
node* sample_nodes; // 含有采样点的同一深度的node集合，用它来扩展出同一层所有节点
int* nodes_num;
std::vector<node*> nodes_d[10]; // 八叉树每层的节点数组
int size_d[10]; // 八叉树每层的节点数量
int tot_size; // 八叉树总节点数

// 假设D < 10
function* key_to_Fo[10];
double* table1[10][10];
double* table2[10][10];
double* table3[10][10];

void make_inner_product_table(int D, double width) {
	// Fo(p)=Fo(x)Fo(y)Fo(z)
	// <Fo'(p), Fo(p)>=<Fo'(x), Fo(x)>*<Fo'(y), Fo(y)>*<Fo'(z), Fo(z)>
	// <△Fo'(p), Fo(p)>=<d2[Fo'(x)]/d2x, Fo(x)>*<Fo'(y), Fo(y)>*<Fo'(z), Fo(z)>+...轮换对称求和...
	// <▽Fo'(p), Fo(p)>的x分量=<d[Fo'(x)]/dx, Fo(x)>*<Fo'(y), Fo(y)>*<Fo'(z), Fo(z)>，y分量与z分量类似
	// 因此只需制作<Fo'(x), Fo(x)>、<d[Fo'(x)]/dx, Fo(x)>、<d2[Fo'(x)]/d2x, Fo(x)>三张内积表，分别对应table1,table2,table3
	// 如果只是为了知道Fo(x)，我们其实不需要节点完整编号，只要知道center_x与width
	// 我们只要知道x每次的走向，就能算出center_x
	// 我们只要知道x所在节点的深度，就能算出width
	// 论文2中说只要用Shuffled xyz Key的x-bits来查询，但是我觉得这样好像缺失了深度信息，因为查询索引是整数而不是字符串，我们不知道前导零或后导零是x-bit还是填充
	// 所以我把深度纳入表的索引中
	for (int d = 0;d <= D;d++) {
		void* p = operator new[]((1 << d) * sizeof(function));
		key_to_Fo[d] = static_cast<function*>(p);
		for (int i = 0;i < (1 << d);i++) {
			new(&key_to_Fo[d][i]) function(3, 2);
		}
		for (unsigned int key = 0;key < (1 << d);key++) {
			x_bits xbits;
			xbits.init(key, d);
			double x = xbits.get_center_x(width);
			key_to_Fo[d][key] = std::move(get_Fo(x, width / (1 << d)));
		}
	}
	for (int d1 = 0;d1 <= D;d1++) {
		for (int d2 = 0;d2 <= D;d2++) {
			int len1 = (1 << d1), len2 = (1 << d2);
			table1[d1][d2] = new double[len1 * len2]();
			table2[d1][d2] = new double[len1 * len2]();
			table3[d1][d2] = new double[len1 * len2]();
			for (unsigned int key1 = 0;key1 < len1;key1++) {
				function& Fo1 = key_to_Fo[d1][key1];
				function dFo1(std::move(differential(Fo1)));
				function ddFo1(std::move(differential(dFo1)));
				for (unsigned int key2 = 0;key2 < len2;key2++) {
					function& Fo2 = key_to_Fo[d2][key2];
					table1[d1][d2][key1 * len2 + key2] = inner_product(Fo1, Fo2);
					table2[d1][d2][key1 * len2 + key2] = inner_product(dFo1, Fo2);
					table3[d1][d2][key1 * len2 + key2] = inner_product(ddFo1, Fo2);
				}
			}
		}
	}
}

int pre_size[10];

struct divergence :public operation {
	void operator()(node* a, node* b) override {
		if (b->depth != D) return;
		normal u;
		u.x = table2[b->depth][a->depth][b->key.extract(0).value * (1 << a->depth) + a->key.extract(0).value] * table1[b->depth][a->depth][b->key.extract(1).value * (1 << a->depth) + a->key.extract(1).value] * table1[b->depth][a->depth][b->key.extract(2).value * (1 << a->depth) + a->key.extract(2).value];
		u.y = table2[b->depth][a->depth][b->key.extract(1).value * (1 << a->depth) + a->key.extract(1).value] * table1[b->depth][a->depth][b->key.extract(0).value * (1 << a->depth) + a->key.extract(0).value] * table1[b->depth][a->depth][b->key.extract(2).value * (1 << a->depth) + a->key.extract(2).value];
		u.z = table2[b->depth][a->depth][b->key.extract(2).value * (1 << a->depth) + a->key.extract(2).value] * table1[b->depth][a->depth][b->key.extract(0).value * (1 << a->depth) + a->key.extract(0).value] * table1[b->depth][a->depth][b->key.extract(1).value * (1 << a->depth) + a->key.extract(1).value];
		u *= -1;
		a->b += b->v * u;
	}
};

struct laplacian :public operation {
	std::vector<Eigen::Triplet<double>>* tripletList = nullptr;
	int d = 0;
	void operator()(node* a, node* b) override {
		if (b->depth != d) return;
		int idx1 = a->idx_node - pre_size[a->depth];
		int idx2 = b->idx_node - pre_size[b->depth];
		double value = 0;
		value += table3[a->depth][b->depth][a->key.extract(0).value * (1 << b->depth) + b->key.extract(0).value] * table1[a->depth][b->depth][a->key.extract(1).value * (1 << b->depth) + b->key.extract(1).value] * table1[a->depth][b->depth][a->key.extract(2).value * (1 << b->depth) + b->key.extract(2).value];
		value += table3[a->depth][b->depth][a->key.extract(1).value * (1 << b->depth) + b->key.extract(1).value] * table1[a->depth][b->depth][a->key.extract(0).value * (1 << b->depth) + b->key.extract(0).value] * table1[a->depth][b->depth][a->key.extract(2).value * (1 << b->depth) + b->key.extract(2).value];
		value += table3[a->depth][b->depth][a->key.extract(2).value * (1 << b->depth) + b->key.extract(2).value] * table1[a->depth][b->depth][a->key.extract(1).value * (1 << b->depth) + b->key.extract(1).value] * table1[a->depth][b->depth][a->key.extract(0).value * (1 << b->depth) + b->key.extract(0).value];
		value = -value;
		if (value != 0) {
			tripletList->emplace_back(idx1, idx2, value);
		}
	}
};

Eigen::SparseMatrix<double> construct_laplacian_matrix(int d, int d2) {
	assert(d >= d2);
	Eigen::SparseMatrix<double> L(size_d[d], size_d[d2]);
	std::vector<Eigen::Triplet<double>> tripletList;
	// 需要注意的是这里按论文2的做法，相当于把基函数支撑集截断到[-1,1]，如果按[-1.5,1.5]来做需要枚举邻居的邻居，误差应该不大
	/*laplacian lf;
	lf.d = d2;
	lf.tripletList = &tripletList;*/
	for (int i = 0;i < size_d[d];i++) {
		node& o = *nodes_d[d][i];
		//o.process(nodes_d[0][0], &lf);
		int idx = o.idx_node - pre_size[d];
		node* pa = &o;
		for (int j = 0;j < d - d2;j++) pa = pa->parent;
		for (int j = 0;j < 27;j++) {
			if (pa->neighbors[j] != nullptr) {
				node& o2 = *pa->neighbors[j];
				int idx2 = o2.idx_node - pre_size[d2];
				double value = 0;
				value += table3[d][d2][o.key.extract(0).value * (1 << d2) + o2.key.extract(0).value] * table1[d][d2][o.key.extract(1).value * (1 << d2) + o2.key.extract(1).value] * table1[d][d2][o.key.extract(2).value * (1 << d2) + o2.key.extract(2).value];
				value += table3[d][d2][o.key.extract(1).value * (1 << d2) + o2.key.extract(1).value] * table1[d][d2][o.key.extract(0).value * (1 << d2) + o2.key.extract(0).value] * table1[d][d2][o.key.extract(2).value * (1 << d2) + o2.key.extract(2).value];
				value += table3[d][d2][o.key.extract(2).value * (1 << d2) + o2.key.extract(2).value] * table1[d][d2][o.key.extract(1).value * (1 << d2) + o2.key.extract(1).value] * table1[d][d2][o.key.extract(0).value * (1 << d2) + o2.key.extract(0).value];
				value = -value;
				if (value != 0) {
					tripletList.emplace_back(idx, idx2, value);
				}
			}
		}
		// 暴力做法
		/*for (int i2 = 0;i2 < size_d[d2];i2++) {
			node& o2 = *nodes_d[d2][i2];
			int idx2 = o2.idx_node - pre_size[d2];
			double value = 0;
			value += table3[d][d2][o.key.extract(0).value * (1 << d2) + o2.key.extract(0).value] * table1[d][d2][o.key.extract(1).value * (1 << d2) + o2.key.extract(1).value] * table1[d][d2][o.key.extract(2).value * (1 << d2) + o2.key.extract(2).value];
			value += table3[d][d2][o.key.extract(1).value * (1 << d2) + o2.key.extract(1).value] * table1[d][d2][o.key.extract(0).value * (1 << d2) + o2.key.extract(0).value] * table1[d][d2][o.key.extract(2).value * (1 << d2) + o2.key.extract(2).value];
			value += table3[d][d2][o.key.extract(2).value * (1 << d2) + o2.key.extract(2).value] * table1[d][d2][o.key.extract(1).value * (1 << d2) + o2.key.extract(1).value] * table1[d][d2][o.key.extract(0).value * (1 << d2) + o2.key.extract(0).value];
			value = -value;
			if (value != 0) {
				tripletList.emplace_back(idx, idx2, value);
			}
		}*/
	}
	L.setFromTriplets(tripletList.begin(), tripletList.end());
	return L;
}
Eigen::VectorXd solve_conjugate_gradient(const Eigen::SparseMatrix<double>& M, const Eigen::VectorXd& b, int iters) {
	const double eps = 1e-6;
	Eigen::VectorXd x, d, r, Md;
	double alpha, beta, r_dot_r, b_dot_b;
	x.resize(b.size());
	x.setZero();
	d = r = b - M * x;
	r_dot_r = r.dot(r);
	b_dot_b = b.dot(b);
	if (b_dot_b < eps) {
		return x;
	}
	for (int i = 0;i < iters;i++) {
		Md = M * d;
		double temp = d.dot(Md);
		if (fabs(temp) < eps) break;
		alpha = r_dot_r / temp;
		r -= alpha * Md;
		temp = r.dot(r);
		if (temp / b_dot_b < eps) break;
		beta = temp / r_dot_r;
		x += alpha * d;
		if (beta < eps) break;
		r_dot_r = temp;
		d = r + beta * d;
	}
	return x;
}

struct triangle {
	point p[3];
};
struct point_comparator {
	bool operator()(const point& p1, const point& p2) const {
		const double threshold = 1e-8;
		if (fabs(p1.x - p2.x) >= threshold) {
			return p1.x < p2.x;
		}
		if (fabs(p1.y - p2.y) >= threshold) {
			return p1.y < p2.y;
		}
		if (fabs(p1.z - p2.z) >= threshold) {
			return p1.z < p2.z;
		}
		return false;
	}
};
std::map<point, double, point_comparator> value_at_point;
std::map<point, int, point_comparator> triangle_vertices_idx;
point* triangle_vertices;
int marching_cube_table[256][15] = {
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1},
	{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1},
	{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1},
	{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1},
	{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1},
	{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1},
	{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1},
	{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1},
	{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1},
	{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1},
	{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1},
	{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1},
	{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1},
	{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1},
	{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1},
	{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1},
	{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1},
	{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1},
	{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1},
	{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1},
	{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1},
	{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1},
	{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1},
	{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1},
	{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1},
	{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1},
	{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1},
	{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1},
	{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0},
	{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0},
	{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1},
	{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1},
	{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1},
	{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1},
	{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1},
	{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1},
	{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1},
	{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1},
	{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1},
	{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1},
	{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1},
	{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9},
	{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1},
	{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1},
	{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6},
	{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1},
	{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11},
	{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7},
	{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1},
	{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1},
	{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1},
	{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1},
	{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1},
	{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1},
	{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1},
	{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1},
	{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1},
	{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1},
	{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1},
	{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1},
	{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1},
	{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1},
	{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1},
	{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1},
	{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1},
	{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1},
	{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9},
	{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1},
	{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1},
	{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7},
	{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11},
	{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1},
	{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6},
	{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1},
	{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1},
	{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1},
	{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1},
	{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1},
	{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1},
	{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1},
	{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1},
	{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1},
	{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1},
	{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1},
	{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1},
	{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1},
	{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1},
	{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1},
	{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3},
	{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1},
	{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1},
	{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1},
	{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1},
	{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1},
	{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3},
	{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1},
	{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1},
	{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1},
	{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1},
	{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1},
	{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6},
	{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1},
	{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1},
	{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8},
	{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1},
	{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4},
	{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10},
	{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1},
	{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1},
	{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1},
	{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1},
	{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10},
	{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5},
	{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1},
	{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1},
	{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1},
	{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8},
	{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6},
	{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1},
	{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1},
	{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1},
	{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1},
	{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1},
	{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1},
	{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2},
	{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1},
	{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1},
	{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1},
	{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2},
	{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1},
	{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1},
	{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1},
	{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1},
	{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1},
	{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4},
	{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1},
	{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11},
	{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5},
	{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1},
	{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1},
	{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9},
	{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1},
	{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1},
	{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1},
	{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1},
	{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4},
	{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1},
	{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3},
	{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1},
	{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1},
	{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1},
	{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7},
	{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10},
	{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1},
	{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1},
	{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1},
	{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1},
	{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1},
	{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1},
	{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};
double evaluate_point(node* cur, point p) {
	if (value_at_point.find(p) != value_at_point.end()) {
		return value_at_point[p];
	}
	std::stack<node*> s;
	s.push(cur);
	double value = 0;
	while (!s.empty()) {
		cur = s.top();
		s.pop();
		double weight = F((p.x - cur->center.x) / cur->width) * F((p.y - cur->center.y) / cur->width) * F((p.z - cur->center.z) / cur->width);
		value += weight * solution[cur->idx_node];
		if (cur->has_children) {
			for (int i = 0;i < 8;i++) {
				node& o = *cur->children[i];
				if (o.idx_node != -1 && fabs(p.x - o.center.x) < 1.5 * o.width && fabs(p.y - o.center.y) < 1.5 * o.width && fabs(p.z - o.center.z) < 1.5 * o.width) {
					s.push(&o);
				}
			}
		}
	}
	value_at_point[p] = value;
	return value;
}
std::map<point, int, point_comparator> edge_root_number[3][10], face_root_number[3][10];
double iso_value = 0;
std::set<point, point_comparator> vis;
void extend_finest_nodes() {
	int edges[12][2] = { {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7} };
	node* root = nodes_d[0][0];
	double width = nodes_d[D][0]->width;
	std::stack<point> s;
	for (int i = 0;i < size_d[D];i++) {
		node& o = *nodes_d[D][i];
		s.push(o.center);
	}
	point node_center, new_node_center;
	while (!s.empty()) {
		node_center = s.top();
		s.pop();
		if (vis.find(node_center) == vis.end()) vis.insert(node_center);
		else continue;
		point p[8];
		// 注意MC算法中顶点编号与八叉树算法中顶点编号不同，这里的编号参考https://github.com/Goodhao/marching-cube/blob/main/encode.png
		p[0] = node_center + point{-width / 2, -width / 2, +width / 2};
		p[1] = node_center + point{-width / 2, +width / 2, +width / 2};
		p[2] = node_center + point{-width / 2, +width / 2, -width / 2};
		p[3] = node_center + point{-width / 2, -width / 2, -width / 2};
		p[4] = node_center + point{+width / 2, -width / 2, +width / 2};
		p[5] = node_center + point{+width / 2, +width / 2, +width / 2};
		p[6] = node_center + point{+width / 2, +width / 2, -width / 2};
		p[7] = node_center + point{+width / 2, -width / 2, -width / 2};
		for (int e = 0;e < 12;e++) {
			int idx1 = edges[e][0];
			int idx2 = edges[e][1];
			point center = (p[idx1] + p[idx2]) / 2;
			int dir = 0;
			if (e == 8 || e == 9 || e == 10 || e == 11) dir = 0; // x轴方向
			if (e == 0 || e == 2 || e == 4 || e == 6) dir = 1; // y轴方向
			if (e == 1 || e == 3 || e == 5 || e == 7) dir = 2; // z轴方向
			if (edge_root_number[dir][D].find(center) == edge_root_number[dir][D].end()) {
				double v1 = evaluate_point(root, p[idx1]) - iso_value;
				double v2 = evaluate_point(root, p[idx2]) - iso_value;
				if (v1 < 0 && v2 > 0 || v1 > 0 && v2 < 0) {
					edge_root_number[dir][D][center] = 1;
				}
			}
		}
		for (int e = 0;e < 12;e++) {
			int idx1 = edges[e][0];
			int idx2 = edges[e][1];
			point center = (p[idx1] + p[idx2]) / 2;
			int dir = 0;
			if (e == 8 || e == 9 || e == 10 || e == 11) dir = 0; // x轴方向
			if (e == 0 || e == 2 || e == 4 || e == 6) dir = 1; // y轴方向
			if (e == 1 || e == 3 || e == 5 || e == 7) dir = 2; // z轴方向
			if (edge_root_number[dir][D].find(center) != edge_root_number[dir][D].end()) {
				// 枚举边的3个虚拟邻居
				new_node_center = 2 * (center - node_center) + node_center;
				if (vis.find(new_node_center) == vis.end()) {
					s.push(new_node_center);
				}
				if (dir == 0) {
					new_node_center = { node_center.x, node_center.y, 2 * (center.z - node_center.z) + node_center.z };
					if (vis.find(new_node_center) == vis.end()) {
						s.push(new_node_center);
					}
					new_node_center = { node_center.x, 2 * (center.y - node_center.y) + node_center.y, node_center.z };
					if (vis.find(new_node_center) == vis.end()) {
						s.push(new_node_center);
					}
				}
				else if (dir == 1) {
					new_node_center = { node_center.x, node_center.y, 2 * (center.z - node_center.z) + node_center.z };
					if (vis.find(new_node_center) == vis.end()) {
						s.push(new_node_center);
					}
					new_node_center = { 2 * (center.x - node_center.x) + node_center.x, node_center.y, node_center.z };
					if (vis.find(new_node_center) == vis.end()) {
						s.push(new_node_center);
					}
				}
				else if (dir == 2) {
					new_node_center = { node_center.x, 2 * (center.y - node_center.y) + node_center.y, node_center.z };
					if (vis.find(new_node_center) == vis.end()) {
						s.push(new_node_center);
					}
					new_node_center = { 2 * (center.x - node_center.x) + node_center.x, node_center.y, node_center.z };
					if (vis.find(new_node_center) == vis.end()) {
						s.push(new_node_center);
					}
				}
			}
		}
	}
}
int edge_root_count(int dir, int depth, point center) {
	// {边方向、深度、边的中点}（在所有合法的边中）唯一确定一条边
	auto it = edge_root_number[dir][depth].find(center);
	if (it != edge_root_number[dir][depth].end()) {
		return it->second;
	}
	else if (depth == D) {
		return 0;
	}
	else {
		double width = 1.0 / (1 << depth);
		int cnt = 0;
		point new_center = center;
		if (dir == 0) {
			new_center.x = center.x + width / 4;
			cnt += edge_root_count(dir, depth + 1, new_center);
			new_center.x = center.x - width / 4;
			cnt += edge_root_count(dir, depth + 1, new_center);
		}
		else if (dir == 1) {
			new_center.y = center.y + width / 4;
			cnt += edge_root_count(dir, depth + 1, new_center);
			new_center.y = center.y - width / 4;
			cnt += edge_root_count(dir, depth + 1, new_center);
		}
		else if (dir == 2) {
			new_center.z = center.z + width / 4;
			cnt += edge_root_count(dir, depth + 1, new_center);
			new_center.z = center.z - width / 4;
			cnt += edge_root_count(dir, depth + 1, new_center);
		}
		return edge_root_number[dir][depth][center] = cnt;
	}
}
void compute_edge_root_number() {
	int edges[12][2] = { {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7} };
	node* root = nodes_d[0][0];
	for (int d = D - 1;d >= 0;d--) {
		for (int i = 0;i < size_d[d];i++) {
			node& o = *nodes_d[d][i];
			double width = o.width;
			point p[8];
			// 注意MC算法中顶点编号与八叉树算法中顶点编号不同，这里的编号参考https://github.com/Goodhao/marching-cube/blob/main/encode.png
			p[0] = o.center + point{-width / 2, -width / 2, +width / 2};
			p[1] = o.center + point{-width / 2, +width / 2, +width / 2};
			p[2] = o.center + point{-width / 2, +width / 2, -width / 2};
			p[3] = o.center + point{-width / 2, -width / 2, -width / 2};
			p[4] = o.center + point{+width / 2, -width / 2, +width / 2};
			p[5] = o.center + point{+width / 2, +width / 2, +width / 2};
			p[6] = o.center + point{+width / 2, +width / 2, -width / 2};
			p[7] = o.center + point{+width / 2, -width / 2, -width / 2};
			for (int e = 0;e < 12;e++) {
				int idx1 = edges[e][0];
				int idx2 = edges[e][1];
				point center = (p[idx1] + p[idx2]) / 2;
				int dir = 0;
				if (e == 8 || e == 9 || e == 10 || e == 11) dir = 0; // x轴方向
				if (e == 0 || e == 2 || e == 4 || e == 6) dir = 1; // y轴方向
				if (e == 1 || e == 3 || e == 5 || e == 7) dir = 2; // z轴方向
				edge_root_count(dir, d, center);
			}
		}
	}
}
int face_root_count(int dir, int depth, point center) {
	// {法线方向、深度、面的中心点}（在所有合法的面中）唯一确定一个面
	// 计算面内部的交点个数，不包括边上的交点个数
	auto it = face_root_number[dir][depth].find(center);
	if (it != face_root_number[dir][depth].end()) {
		return it->second;
	}
	else if (depth == D) {
		return 0;
	}
	else {
		double width = 1.0 / (1 << depth);
		int cnt = 0;
		if (dir == 0) {
			point c;
			c = { center.x, center.y + width / 4, center.z + width / 4 };
			cnt += face_root_count(dir, depth + 1, c);
			c = { center.x, center.y + width / 4, center.z - width / 4 };
			cnt += face_root_count(dir, depth + 1, c);
			c = { center.x, center.y - width / 4, center.z + width / 4 };
			cnt += face_root_count(dir, depth + 1, c);
			c = { center.x, center.y - width / 4, center.z - width / 4 };
			cnt += face_root_count(dir, depth + 1, c);

			c = { center.x, center.y - width / 4, center.z };
			cnt += edge_root_count(dir, depth + 1, c);
			c = { center.x, center.y + width / 4, center.z };
			cnt += edge_root_count(dir, depth + 1, c);
			c = { center.x, center.y, center.z - width / 4 };
			cnt += edge_root_count(dir, depth + 1, c);
			c = { center.x, center.y, center.z + width / 4 };
			cnt += edge_root_count(dir, depth + 1, c);
		}
		else if (dir == 1) {
			point c;
			c = { center.x + width / 4, center.y, center.z + width / 4 };
			cnt += face_root_count(dir, depth + 1, c);
			c = { center.x + width / 4, center.y, center.z - width / 4 };
			cnt += face_root_count(dir, depth + 1, c);
			c = { center.x - width / 4, center.y, center.z + width / 4 };
			cnt += face_root_count(dir, depth + 1, c);
			c = { center.x - width / 4, center.y, center.z - width / 4 };
			cnt += face_root_count(dir, depth + 1, c);

			c = { center.x - width / 4, center.y, center.z };
			cnt += edge_root_count(dir, depth + 1, c);
			c = { center.x + width / 4, center.y, center.z };
			cnt += edge_root_count(dir, depth + 1, c);
			c = { center.x, center.y, center.z - width / 4 };
			cnt += edge_root_count(dir, depth + 1, c);
			c = { center.x, center.y, center.z + width / 4 };
			cnt += edge_root_count(dir, depth + 1, c);
		}
		else if (dir == 2) {
			point c;
			c = { center.x + width / 4, center.y + width / 4, center.z };
			cnt += face_root_count(dir, depth + 1, c);
			c = { center.x + width / 4, center.y - width / 4, center.z };
			cnt += face_root_count(dir, depth + 1, c);
			c = { center.x - width / 4, center.y + width / 4, center.z };
			cnt += face_root_count(dir, depth + 1, c);
			c = { center.x - width / 4, center.y - width / 4, center.z };
			cnt += face_root_count(dir, depth + 1, c);

			c = { center.x, center.y - width / 4, center.z };
			cnt += edge_root_count(dir, depth + 1, c);
			c = { center.x, center.y + width / 4, center.z };
			cnt += edge_root_count(dir, depth + 1, c);
			c = { center.x - width / 4, center.y, center.z };
			cnt += edge_root_count(dir, depth + 1, c);
			c = { center.x + width / 4, center.y, center.z };
			cnt += edge_root_count(dir, depth + 1, c);
		}
		return face_root_number[dir][depth][center] = cnt;
	}
}
void compute_face_root_number() {
	int edges[12][2] = { {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7} };
	int faces[6][4] = { {3,0,1,2},{5,4,7,6},{7,8,3,11},{1,9,5,10},{2,11,6,10},{4,9,0,8} };
	node* root = nodes_d[0][0];
	for (int d = D - 1;d >= 0;d--) {
		for (int i = 0;i < size_d[d];i++) {
			node& o = *nodes_d[d][i];
			double width = o.width;
			point p[8];
			// 注意MC算法中顶点编号与八叉树算法中顶点编号不同，这里的编号参考https://github.com/Goodhao/marching-cube/blob/main/encode.png
			p[0] = o.center + point{-width / 2, -width / 2, +width / 2};
			p[1] = o.center + point{-width / 2, +width / 2, +width / 2};
			p[2] = o.center + point{-width / 2, +width / 2, -width / 2};
			p[3] = o.center + point{-width / 2, -width / 2, -width / 2};
			p[4] = o.center + point{+width / 2, -width / 2, +width / 2};
			p[5] = o.center + point{+width / 2, +width / 2, +width / 2};
			p[6] = o.center + point{+width / 2, +width / 2, -width / 2};
			p[7] = o.center + point{+width / 2, -width / 2, -width / 2};
			for (int f = 0;f < 6;f++) {
				point center;
				for (int j = 0;j < 4;j++) {
					center = center + p[edges[faces[f][j]][0]] + p[edges[faces[f][j]][1]];
				}
				center = center / 8;
				int dir = 0;
				if (f == 0 || f == 1) dir = 0; // x轴法向
				if (f == 2 || f == 3) dir = 1; // y轴法向
				if (f == 4 || f == 5) dir = 2; // z轴法向
				face_root_count(dir, d, center);
			}
		}
	}
}
void marching_cube() {
	node* root = nodes_d[0][0];

	double sum = 0;
	double val = 0;
	for (int i = 0;i < size_d[D];i++) {
		node& o = *nodes_d[D][i];
		/*val += o.sampling_density * evaluate_point(root, o.center);
		sum += o.sampling_density;*/
		double w = sqrt(o.v * o.v);
		val += w * evaluate_point(root, o.center);
		sum += w;
	}
	iso_value = val / sum;
	/*for (int i = 0;i < n;i++) {
		val += evaluate_point(root, a[i]);
	}
	iso_value = val / n;
	value_at_point.clear();*/
	std::cout << "等值面的值为：" << iso_value << std::endl;

	extend_finest_nodes();
	std::cout << "最细分辨率立方体上的交点扩展完毕" << std::endl;

	/*std::ofstream file2;
	file2.open("V.txt");
	for (auto p : vis) {
		file2 << p.x << " " << p.y << " " << p.z << " " << 0 << " " << 0 << " " << 0 << std::endl;
	}
	file2.close();*/

	compute_edge_root_number();
	compute_face_root_number();
	std::cout << "等值面交点预处理完毕" << std::endl;
	int edges[12][2] = { {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7} };
	int faces[6][4] = { {3,0,1,2},{5,4,7,6},{7,8,3,11},{1,9,5,10},{2,11,6,10},{4,9,0,8} };
	for (int d = 0;d < D;d++) { // 跳过d=D，因为最多细分到D
		for (int i = 0;i < size_d[d];i++) {
			node& o = *nodes_d[d][i];
			if (o.has_children) continue; // 跳过非叶节点
			double width = o.width;
			point p[8];
			p[0] = o.center + point{-width / 2, -width / 2, +width / 2};
			p[1] = o.center + point{-width / 2, +width / 2, +width / 2};
			p[2] = o.center + point{-width / 2, +width / 2, -width / 2};
			p[3] = o.center + point{-width / 2, -width / 2, -width / 2};
			p[4] = o.center + point{+width / 2, -width / 2, +width / 2};
			p[5] = o.center + point{+width / 2, +width / 2, +width / 2};
			p[6] = o.center + point{+width / 2, +width / 2, -width / 2};
			p[7] = o.center + point{+width / 2, -width / 2, -width / 2};
			bool subdivide = false;
			//if (o.idx_node == -1) subdivide = true;
			if (!subdivide) for (int e = 0;e < 12;e++) {
				int idx1 = edges[e][0];
				int idx2 = edges[e][1];
				point center = (p[idx1] + p[idx2]) / 2;
				int dir = 0;
				if (e == 8 || e == 9 || e == 10 || e == 11) dir = 0; // x轴方向
				if (e == 0 || e == 2 || e == 4 || e == 6) dir = 1; // y轴方向
				if (e == 1 || e == 3 || e == 5 || e == 7) dir = 2; // z轴方向
				if (edge_root_count(dir, o.depth, center) > 0) { // 注意这里与论文1实现方式不同，论文1是边交点超过1个才细分，但是那样需要对不同分辨率都构造三角形，而我只对最细分辨率构造三角形，所以粗分辨率的边交点数即使=1也得细分
					subdivide = true;
					break;
				}
			}
			if (!subdivide) for (int f = 0;f < 6;f++) {
				int idx1 = faces[f][0];
				int idx2 = faces[f][1];
				int idx3 = faces[f][2];
				int idx4 = faces[f][3];
				point center;
				for (int j = 0;j < 4;j++) {
					center = center + p[edges[faces[f][j]][0]] + p[edges[faces[f][j]][1]];
				}
				center = center / 8;
				int dir = 0;
				if (f == 0 || f == 1) dir = 0; // x轴法向
				if (f == 2 || f == 3) dir = 1; // y轴法向
				if (f == 4 || f == 5) dir = 2; // z轴法向
				if (face_root_count(dir, o.depth, center) > 0) {
					subdivide = true;
					break;
				}
			}
			if (subdivide) {
				o.init_children();
				for (int j = 0;j < 8;j++) {
					node& o2 = *o.children[j];
					size_d[o2.depth]++;
					nodes_d[o2.depth].push_back(&o2);
				}
			}
		}
	}

	std::vector<triangle> triangles;
	for (int d = D;d <= D;d++) {
		for (int i = 0;i < size_d[d];i++) {
			node& o = *nodes_d[d][i];
			if (o.has_children) continue; // 跳过非叶节点
			double width = o.width;
			point p[8];
			p[0] = o.center + point{-width / 2, -width / 2, +width / 2};
			p[1] = o.center + point{-width / 2, +width / 2, +width / 2};
			p[2] = o.center + point{-width / 2, +width / 2, -width / 2};
			p[3] = o.center + point{-width / 2, -width / 2, -width / 2};
			p[4] = o.center + point{+width / 2, -width / 2, +width / 2};
			p[5] = o.center + point{+width / 2, +width / 2, +width / 2};
			p[6] = o.center + point{+width / 2, +width / 2, -width / 2};
			p[7] = o.center + point{+width / 2, -width / 2, -width / 2};

			int idx = 0;
			for (int j = 7;j >= 0;j--) {
				idx <<= 1;
				idx += (evaluate_point(root, p[j]) - iso_value > 0);
			}
			for (int j = 0;j < 5;j++) {
				triangle t;
				bool computed = false;
				for (int k = 0;k < 3;k++) {
					int e = marching_cube_table[idx][j * 3 + k];
					if (e == -1) {
						computed = false;
						break;
					}
					std::vector<int> v;
					if (e == 0) {
						v = { 0, 1 };
					}
					else if (e == 1) {
						v = { 1, 2 };
					}
					else if (e == 2) {
						v = { 2, 3 };
					}
					else if (e == 3) {
						v = { 3, 0 };
					}
					else if (e == 4) {
						v = { 4, 5 };
					}
					else if (e == 5) {
						v = { 5, 6 };
					}
					else if (e == 6) {
						v = { 6, 7 };
					}
					else if (e == 7) {
						v = { 7, 4 };
					}
					else if (e == 8) {
						v = { 0, 4 };
					}
					else if (e == 9) {
						v = { 1, 5 };
					}
					else if (e == 10) {
						v = { 2, 6 };
					}
					else if (e == 11) {
						v = { 3, 7 };
					}
					double k0, k1;
					double v0, v1;
					v0 = evaluate_point(root, p[v[0]]) - iso_value;
					v1 = evaluate_point(root, p[v[1]]) - iso_value;
					assert(v0 > 0 && v1 < 0 || v0 < 0 && v1 > 0);
					k0 = -v1 / (v0 - v1);
					k1 = v0 / (v0 - v1);
					assert(0 <= k0 && k0 <= 1);
					assert(0 <= k1 && k1 <= 1);
					t.p[k] = p[v[0]] * k0 + p[v[1]] * k1;
					computed = true;
				}
				if (computed) {
					triangles.push_back(t);
				}
			}
		}
	}

	std::ofstream file;
	file.open("triangles.txt");
	for (const auto& triangle : triangles) {
		for (int i = 0;i < 3;i++) {
			const point& p = triangle.p[i];
			if (triangle_vertices_idx.find(p) == triangle_vertices_idx.end()) {
				int idx = triangle_vertices_idx.size();
				triangle_vertices_idx[p] = idx;
			}
		}
	}
	triangle_vertices = new point[triangle_vertices_idx.size()]();
	for (const auto& pair : triangle_vertices_idx) {
		triangle_vertices[pair.second] = pair.first;
	}
	file << triangle_vertices_idx.size() << std::endl;
	for (int i = 0;i < triangle_vertices_idx.size();i++) {
		point& p = triangle_vertices[i];
		file << p.x << " " << p.y << " " << p.z << std::endl;
	}
	for (const auto& triangle : triangles) {
		for (int i = 0;i < 3;i++) {
			const point& p = triangle.p[i];
			file << triangle_vertices_idx[triangle.p[i]] << " ";
		}
		file << std::endl;
	}
}
int main() {
	// 论文1：Poisson Surface Reconstruction
	// 论文2：Data-Parallel Octrees for Surface Reconstruction


	// 第一步：生成八叉树
	std::ifstream point_cloud;
	point_cloud.open("horse.txt");
	//point_cloud.open("sphere_points_and_normals.txt");
	//point_cloud.open("cube.txt");
	//point_cloud.open("plane_points_and_normals.txt");
	//point_cloud.open("small_plane_points_and_normals.txt");
	point_cloud >> n;
	a = new point[n]();
	va = new normal[n]();
	
	double min_x=1e10, max_x=-1e10, min_y=1e10, max_y=-1e10, min_z=1e10, max_z=-1e10;
	for (int i = 0;i < n;i++) {
		double x, y, z, vx, vy, vz;
		point_cloud >> x >> y >> z >> vx >> vy >> vz;
		/*point_cloud >> x >> y >> z;
		vx = vy = vz = 0;*/
		a[i] = point{ x,y,z };
		va[i] = normal{ vx,vy,vz };
		min_x = std::min(min_x, x);
		min_y = std::min(min_y, y);
		min_z = std::min(min_z, z);
		max_x = std::max(max_x, x);
		max_y = std::max(max_y, y);
		max_z = std::max(max_z, z);
	}
	point_cloud.close();

	point center;
	center.x = (min_x + max_x) / 2;
	center.y = (min_y + max_y) / 2;
	center.z = (min_z + max_z) / 2;
	D = 8;
	double scale_ratio = 1.25;
	double width = std::max(max_x - min_x, std::max(max_y - min_y, max_z - min_z));
	for (int i = 0;i < n;i++) {
		a[i].x -= center.x;
		a[i].y -= center.y;
		a[i].z -= center.z;
	}
	for (int i = 0;i < n;i++) {
		a[i].x /= width * scale_ratio;
		a[i].y /= width * scale_ratio;
		a[i].z /= width * scale_ratio;
		assert(-0.5 <= a[i].x && a[i].x <= 0.5);
		assert(-0.5 <= a[i].y && a[i].y <= 0.5);
		assert(-0.5 <= a[i].z && a[i].z <= 0.5);
	}
	width = 1;
	center.x = center.y = center.z = 0; // 把中心放在(0,0,0)处

	codes = new code[n]();
	for (int i = 0;i < n;i++) {
		codes[i].idx = i;
		codes[i].key.compute(i, D, width);
	}
	std::sort(codes, codes + n);
	sorted_a = new point[n]();
	sorted_va = new normal[n]();
	for (int i = 0; i < n;i++) {
		int idx = codes[i].idx;
		sorted_a[i] = a[idx];
		sorted_va[i] = va[idx];
	}
	delete[] a;
	delete[] va;
	a = sorted_a;
	va = sorted_va;
	int m = 0;
	for (int i = 0;i < n;i++) {
		if (i == 0 || !(codes[i].key == codes[i - 1].key)) {
			++m;
		}
	}
	sample_nodes = new node[m]();
	nodes_num = new int[m]();
	int j = -1;
	for (int i = 0;i < n;i++) {
		if (i == 0 || !(codes[i].key == codes[i - 1].key)) {
			++j;
			node& o = sample_nodes[j];
			o.depth = D;
			o.width = width / (1 << D);
			o.key = codes[i].key;
			o.idx_p = i;
			o.cnt_p = 1;
			o.compute_center();
		}
		else {
			sample_nodes[j].cnt_p++;
		}
	}
	delete[] codes;
	for (int i = 0;i < m;i++) {
		if (i == 0 || sample_nodes[i].key.get_parent_key() != sample_nodes[i - 1].key.get_parent_key()) {
			nodes_num[i] = 8;
		}
		else {
			nodes_num[i] = 0;
		}
	}
	for (int i = 1;i < m;i++) nodes_num[i] += nodes_num[i - 1];
	int tot = nodes_num[m - 1];
	size_d[D] = tot;
	nodes_d[D].resize(tot);
	for (int i = 0;i < tot;i++) {
		nodes_d[D][i] = new node();
	}
	for (int i = 0;i < m;i++) {
		int branch_idx = sample_nodes[i].key.get_branch_key();
		delete nodes_d[D][(nodes_num[i] - 8) + branch_idx];
		nodes_d[D][(nodes_num[i] - 8) + branch_idx] = &sample_nodes[i];
	}
	for (int i = 0;i < m;i++) {
		if (i > 0 && nodes_num[i] == nodes_num[i - 1]) continue;
		unsigned int key = (sample_nodes[i].key.get_parent_key() << 3);
		for (int j = 0;j < 8;j++) {
			node& o = *nodes_d[D][(nodes_num[i] - 8) + j];
			if (o.cnt_p == 0) {
				o.depth = D;
				o.width = width / (1 << D);
				o.key.init(key + j, D);
				o.compute_center();
			}
			o.cnt_o = 1;
		}
	}
	for (int i = D;i >= 1;i--) {
		// 已知nodes_d[i]，去构建nodes_d[i-1]
		m = tot / 8;
		sample_nodes = new node[m]();
		for (int j = 0;j < m;j++) {
			node& o = sample_nodes[j];
			o.depth = i - 1;
			o.width = width / (1 << o.depth);
			o.key.init(nodes_d[i][j * 8]->key.get_parent_key(), o.depth);
			o.compute_center();
			o.has_children = true;
			for (int k = 0;k < 8;k++) {
				node& u = *nodes_d[i][j * 8 + k];
				o.children[k] = &u;
				u.parent = &o;
				if (u.cnt_p > 0) {
					if (o.cnt_p == 0) {
						o.idx_p = u.idx_p;
					}
					o.cnt_p += u.cnt_p;
				}
			}
		}
		for (int j = 0;j < m;j++) {
			if (j == 0 || sample_nodes[j].key.get_parent_key() != sample_nodes[j - 1].key.get_parent_key()) {
				nodes_num[j] = 8;
			}
			else {
				nodes_num[j] = 0;
			}
		}
		for (int j = 1;j < m;j++) nodes_num[j] += nodes_num[j - 1];
		tot = nodes_num[m - 1];
		if (i == 1) tot = 1;
		size_d[i - 1] = tot;
		nodes_d[i - 1].resize(tot);
		for (int j = 0;j < tot;j++) {
			nodes_d[i - 1][j] = new node();
		}
		for (int j = 0;j < m;j++) {
			int branch_idx = sample_nodes[j].key.get_branch_key();
			delete nodes_d[i - 1][(nodes_num[j] - 8) + branch_idx];
			nodes_d[i - 1][(nodes_num[j] - 8) + branch_idx] = &sample_nodes[j];
		}
		if (i > 1) for (int j = 0;j < m;j++) {
			if (j > 0 && nodes_num[j] == nodes_num[j - 1]) continue;
			unsigned int key = (sample_nodes[j].key.get_parent_key() << 3);
			for (int k = 0;k < 8;k++) {
				node& o = *nodes_d[i - 1][(nodes_num[j] - 8) + k];
				if (o.cnt_p == 0) {
					o.depth = i - 1;
					o.width = width / (1 << o.depth);
					o.key.init(key + k, o.depth);
					o.compute_center();
				}
			}
		}
	}
	tot_size = 0;
	for (int d = 0;d <= D;d++) {
		for (int i = 0;i < size_d[d];i++) {
			node& o = *nodes_d[d][i];
			o.idx_node = tot_size++;
			o.naive_compute_neighbors();
		}
	}
	for (int d = D;d >= 0;d--) {
		for (int i = 0;i < size_d[d];i++) {
			node& o = *nodes_d[d][i];
			if (d == D) {
				o.idx_o = i;
				o.cnt_o = 1;
			}
			else {
				if (o.has_children) {
					for (int j = 0;j < 8;j++) {
						node& o2 = *o.children[j];
						if (o.idx_o == -1) o.idx_o = o2.idx_o;
						o.cnt_o += o2.cnt_o;
					}
				}
			}
		}
	}
	std::cout << "第一步：八叉树生成完毕" << std::endl;
	std::cout << "节点个数：" << tot_size << std::endl;

	std::ofstream file2;
	file2.open("V.txt");
	for (int i = 0;i < n;i++) {
		point p = a[i];
		file2 << p.x << " " << p.y << " " << p.z << " " << 0 << " " << 0 << " " << 0 << std::endl;
	}
	file2.close();

	// 第二步：构造F
	// F是盒式滤波器B的卷积，可以手算出表达式
	// 论文1的设置，把三个B卷积，得到支撑集为[-1.5,1.5]的函数
	// -1.5<x<=-0.5, F(x)=1.125+1.5x+0.5x^2
	// -0.5<x<=0.5, F(x)=0.75-x^2
	// 0.5<x<=1.5, F(x)=1.125-1.5x+0.5x^2
	// 其他情况, F(x)=0
	F.polys[0].coeffs[0] = 1.125;
	F.polys[0].coeffs[1] = 1.5;
	F.polys[0].coeffs[2] = 0.5;
	F.polys[1].coeffs[0] = 0.75;
	F.polys[1].coeffs[1] = 0;
	F.polys[1].coeffs[2] = -1;
	F.polys[2].coeffs[0] = 1.125;
	F.polys[2].coeffs[1] = -1.5;
	F.polys[2].coeffs[2] = 0.5;
	F.break_points[0] = -1.5;
	F.break_points[1] = -0.5;
	F.break_points[2] = 0.5;
	F.break_points[3] = 1.5;

	// 直接把论文1的函数的定义域重新调整至[-1, 1]，而不是像论文2只把两个B卷积来得到[-1,1]的函数（这个函数并不二次可微，拉普拉斯矩阵没法算？不清楚论文2是如何处理这一点的）
	// 但这样效果不好
	/*F.polys[0].coeffs[0] = 1.125;
	F.polys[0].coeffs[1] = 2.25;
	F.polys[0].coeffs[2] = 1.125;
	F.polys[1].coeffs[0] = 0.75;
	F.polys[1].coeffs[1] = 0;
	F.polys[1].coeffs[2] = -2.25;
	F.polys[2].coeffs[0] = 1.125;
	F.polys[2].coeffs[1] = -2.25;
	F.polys[2].coeffs[2] = 1.125;
	F.break_points[0] = -1;
	F.break_points[1] = -1.0/3.0;
	F.break_points[2] = 1.0/3.0;
	F.break_points[3] = 1;*/
	std::cout << "第二步：标准基函数构造完毕" << std::endl;

	// 第三步：构造向量场V
	// vo = sum_{i} α(o,qi) normal_{qi}，论文1说是三线性插值，与它源码不一致，我们与它源码一样用基函数插值α(o,qi)=Fo(qi)
	// 构造向量场的过程就是先用采样点法向量插值出八叉树节点向量，再用八叉树节点向量作为基函数系数构造出整个向量场（向量值函数）
	for (int d = 0;d <= D;d++) {
		for (int i = 0;i < size_d[d];i++) {
			node& o = *nodes_d[d][i];
			int start_idx = o.idx_p;
			int end_idx = o.idx_p + o.cnt_p - 1;
			for (int idx = start_idx;idx <= end_idx;idx++) {
				point& p = a[idx];
				for (int j = 0;j < 27;j++) {
					if (o.neighbors[j] == nullptr) continue;
					node& o2 = *o.neighbors[j];
					double weight = F((p.x - o2.center.x) / o2.width) * F((p.y - o2.center.y) / o2.width) * F((p.z - o2.center.z) / o2.width);
					//o2.sampling_density += weight;
					o2.sampling_density += 1; // 反而是这种简单粗暴的采样密度估计对于horse.txt来说效果最好
				}
			}
		}
	}
	for (int i = 0;i < size_d[D];i++) {
		node& o = *nodes_d[D][i];
		int start_idx = o.idx_p;
		int end_idx = o.idx_p + o.cnt_p - 1;
		for (int idx = start_idx;idx <= end_idx;idx++) {
			point& p = a[idx];
			double p_sampling_density = 0, area = 0;
			for (int j = 0;j < 27;j++) {
				if (o.neighbors[j] == nullptr) continue;
				node& o2 = *o.neighbors[j];
				double weight = F((p.x - o2.center.x) / o2.width) * F((p.y - o2.center.y) / o2.width) * F((p.z - o2.center.z) / o2.width);
				p_sampling_density += weight * o2.sampling_density;
			}
			assert(p_sampling_density != 0);
			area = 1.0 / p_sampling_density;
			normal n;
			n = va[idx] * area / pow(o.width, 3); // 我不知道除以pow(o.width, 3)的用意，这里是模仿了论文1源码。因为我只把向量在最深层泼溅，所以比例系数都为1/pow(1/(1<<D), 3)，去掉后不会影响
			for (int j = 0;j < 27;j++) {
				if (o.neighbors[j] == nullptr) continue;
				node& o2 = *o.neighbors[j];
				double weight = F((p.x - o2.center.x) / o2.width) * F((p.y - o2.center.y) / o2.width) * F((p.z - o2.center.z) / o2.width);
				o2.v += weight * n;
			}
		}
	}
	std::cout << "第三步：向量场生成完毕" << std::endl;

	// 第四步：计算向量场的散度
	// 需要注意的是这里按论文2的做法，相当于把基函数支撑集截断到[-1,1]，严格按[-1.5,1.5]来做需要枚举邻居的邻居或者从树根一路递归下来，所以其实是有误差的
	make_inner_product_table(D, width);
	//divergence df;
	for (int d = 0;d <= D;d++) {
		for (int i = 0;i < size_d[d];i++) {
			node& o = *nodes_d[d][i];
			//o.process(nodes_d[0][0], &df);
			for (int j = 0;j < 27;j++) {
				if (o.neighbors[j] != nullptr && o.neighbors[j]->idx_o != -1) {
					int start_idx = o.neighbors[j]->idx_o;
					int end_idx = o.neighbors[j]->idx_o + o.neighbors[j]->cnt_o - 1;
					for (int idx = start_idx;idx <= end_idx;idx++) {
						node& o2 = *nodes_d[D][idx];
						normal u;
						u.x = table2[D][d][o2.key.extract(0).value * (1 << d) + o.key.extract(0).value] * table1[D][d][o2.key.extract(1).value * (1 << d) + o.key.extract(1).value] * table1[D][d][o2.key.extract(2).value * (1 << d) + o.key.extract(2).value];
						u.y = table2[D][d][o2.key.extract(1).value * (1 << d) + o.key.extract(1).value] * table1[D][d][o2.key.extract(0).value * (1 << d) + o.key.extract(0).value] * table1[D][d][o2.key.extract(2).value * (1 << d) + o.key.extract(2).value];
						u.z = table2[D][d][o2.key.extract(2).value * (1 << d) + o.key.extract(2).value] * table1[D][d][o2.key.extract(0).value * (1 << d) + o.key.extract(0).value] * table1[D][d][o2.key.extract(1).value * (1 << d) + o.key.extract(1).value];
						u *= -1;
						o.b += o2.v * u;
					}
				}
			}
			// 暴力做法
			/*for (int i2 = 0;i2 < size_d[D];i2++) {
				node& o2 = *nodes_d[D][i2];
				double distance = sqrt(pow(o2.center.x - o.center.x, 2) + pow(o2.center.y - o.center.y, 2) + pow(o2.center.z - o.center.z, 2));
				if (distance >= o.width + o2.width) continue;
				normal u;
				u.x = table2[D][d][o2.key.extract(0).value * (1 << d) + o.key.extract(0).value] * table1[D][d][o2.key.extract(1).value * (1 << d) + o.key.extract(1).value] * table1[D][d][o2.key.extract(2).value * (1 << d) + o.key.extract(2).value];
				u.y = table2[D][d][o2.key.extract(1).value * (1 << d) + o.key.extract(1).value] * table1[D][d][o2.key.extract(0).value * (1 << d) + o.key.extract(0).value] * table1[D][d][o2.key.extract(2).value * (1 << d) + o.key.extract(2).value];
				u.z = table2[D][d][o2.key.extract(2).value * (1 << d) + o.key.extract(2).value] * table1[D][d][o2.key.extract(0).value * (1 << d) + o.key.extract(0).value] * table1[D][d][o2.key.extract(1).value * (1 << d) + o.key.extract(1).value];
				u = (-1) * u;
				o.b += o2.v * u;
			}*/
		}
	}
	std::cout << "第四步：散度计算完毕" << std::endl;

	// 第五步：求解泊松方程
	solution = new double[tot_size]();
	for (int d = 1;d <= D;d++) pre_size[d] = pre_size[d - 1] + size_d[d - 1];


	solution[0] = 0;
	Eigen::VectorXd x[10];
	for (int d = 1;d <= D;d++) {
		Eigen::SparseMatrix<double> L_d = construct_laplacian_matrix(d, d);
		Eigen::VectorXd b(size_d[d]);
		for (int i = 0;i < size_d[d];i++) {
			node& o = *nodes_d[d][i];
			b[i] = o.b;
		}
		for (int d2 = 1;d2 < d;d2++) {
			Eigen::SparseMatrix<double> L_d_d2 = construct_laplacian_matrix(d, d2);
			b -= L_d_d2 * x[d2];
		}
		x[d] = solve_conjugate_gradient(L_d, b, int(std::pow(L_d.rows(), 1.0 / 3.0)));
		//x[d] = solve_conjugate_gradient(L_d, b, int(std::pow(L_d.rows(), 1.0)));
		//x[d] = solve_conjugate_gradient(L_d, b, int(std::pow(L_d.rows(), 0.7)));
		for (int i = 0;i < size_d[d];i++) {
			solution[i + pre_size[d]] = x[d][i];
		}
	}
	/*Eigen::SparseMatrix<double> L(tot_size, tot_size);
	std::vector<Eigen::Triplet<double>> tripletList;
	for (int d = 0;d <= D;d++) {
		for (int i = 0;i < size_d[d];i++) {
			node& o = *nodes_d[d][i];
			int idx = o.idx_node;
			for (int d2 = 0;d2 <= d;d2++) {
				node* pa = &o;
				for (int j = 0;j < d - d2;j++) pa = pa->parent;
				for (int j = 0;j < 27;j++) {
					if (pa->neighbors[j] != nullptr) {
						node& o2 = *pa->neighbors[j];
						int idx2 = o2.idx_node;
						double value = 0;
						value += table3[d][d2][o.key.extract(0).value * (1 << d2) + o2.key.extract(0).value] * table1[d][d2][o.key.extract(1).value * (1 << d2) + o2.key.extract(1).value] * table1[d][d2][o.key.extract(2).value * (1 << d2) + o2.key.extract(2).value];
						value += table3[d][d2][o.key.extract(1).value * (1 << d2) + o2.key.extract(1).value] * table1[d][d2][o.key.extract(0).value * (1 << d2) + o2.key.extract(0).value] * table1[d][d2][o.key.extract(2).value * (1 << d2) + o2.key.extract(2).value];
						value += table3[d][d2][o.key.extract(2).value * (1 << d2) + o2.key.extract(2).value] * table1[d][d2][o.key.extract(1).value * (1 << d2) + o2.key.extract(1).value] * table1[d][d2][o.key.extract(0).value * (1 << d2) + o2.key.extract(0).value];
						value = -value;
						if (value != 0) {
							tripletList.emplace_back(idx, idx2, value);
							if (idx != idx2) {
								tripletList.emplace_back(idx2, idx, -value);
							}
						}
					}
				}
			}
		}
	}
	L.setFromTriplets(tripletList.begin(), tripletList.end());
	Eigen::VectorXd b(tot_size);
	for (int d = 0;d <= D;d++) {
		for (int i = 0;i < size_d[d];i++) {
			node& o = *nodes_d[d][i];
			b[o.idx_node] = o.b;
		}
	}
	Eigen::VectorXd x = solve_conjugate_gradient(L, b, int(std::pow(L.rows(), 0.7)));
	for (int i = 0;i < tot_size;i++) {
		solution[i] = x[i];
	}*/

	std::cout << "第五步：泊松方程求解完毕" << std::endl;

	// 第六步：提取等值面
	marching_cube();
	std::cout << "第六步：等值面提取完毕" << std::endl;
	return 0;
}
