\LoadClass[10pt,twocolumn,letterpaper]{ctexart}
%%%%%%%%% PAPER TYPE  - PLEASE UPDATE FOR FINAL VERSION
% \RequirePackage[review]{source/cvpr}      % To produce the REVIEW version
\RequirePackage{source/cvpr}              % To produce the CAMERA-READY version
%\RequirePackage[pagenumbers]{source/cvpr} % To force page numbers, e.g. for an arXiv version

%宏包
\RequirePackage{xeCJK}
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{geometry}
\RequirePackage{zhnumber}
% \RequirePackage{titlesec}
\RequirePackage{enumitem}
\RequirePackage{listings}
\RequirePackage{xcolor}
\RequirePackage{array}
% \RequirePackage{mathtools}
\RequirePackage{float}
\RequirePackage{indentfirst}
\RequirePackage{braket}
\RequirePackage{zhnumber}
\RequirePackage{multirow}
\RequirePackage{listings}
\RequirePackage{lmodern}
\RequirePackage{amssymb}
\RequirePackage{ulem}
\RequirePackage{accsupp}
\RequirePackage{diagbox}
\RequirePackage{unicode-math}
\RequirePackage{extarrows}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{multicol}
\RequirePackage{supertabular}
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\RequirePackage{amsmath}
\RequirePackage{wrapfig}
\RequirePackage{titletoc}
\RequirePackage{tikz}
% \RequirePackage{pgfplots}
\RequirePackage[pagebackref, breaklinks, hidelinks]{hyperref}
\RequirePackage[capitalize]{cleveref}
\crefname{section}{Sec.}{Secs.}
\Crefname{section}{Section}{Sections}
\Crefname{table}{Table}{Tables}
\crefname{table}{Tab.}{Tabs.}

\usetikzlibrary{positioning,arrows.meta,quotes}
\usetikzlibrary{shapes}
\usetikzlibrary{bayesnet}
\usetikzlibrary{decorations.pathreplacing}
\tikzset{>=stealth}
\definecolor{light_red}{RGB}{209,105,81}
\definecolor{light_green}{RGB}{58,181,75}
\definecolor{light_blue}{RGB}{0,153,228}

\setlength{\headheight}{15pt}

\newtheorem{theorem}{Theorem}[section]
\newtheorem{definition}{Definition}[section]
\numberwithin{equation}{section}

\hypersetup{
	colorlinks=true,%打开颜色显示
	linkcolor=light_blue,%内部链接
	filecolor=magenta,%本地连接
	urlcolor=cyan,%网页链接
	% pdftitle={Overleaf Example},
	% pdfpagemode=FullScreen,
}

%自定义命令
	% 复制代码不带行号
	\newcommand{\emptyaccsupp}[1]{\BeginAccSupp{ActualText={}}#1\EndAccSupp{}}
	%脚注
	\newcommand{\Footnote}[1]{\footnote{\small\kaishu #1}}
	%标红
	\newcommand{\hl}[1]{\textcolor{light_red}{#1}}
	% 定义°为\degree 不需要$, \deg 需要$
	\def\deg{^{\circ}}
	\def\degree{${}^{\circ}$}
	% 在外部打上tikz标记, 在tikz中加上 overlay 可与之互动. \tikzmark[颜色]{名字}{具体文字} 
	\newcommand{\tikzmark}[3][] {\tikz[remember picture, baseline] \node [anchor=base,#1](#2) {#3};}
	% 重定义表格编号
	\renewcommand\tablename{\textbf{Table}}
	% 重定义图片编号
	\renewcommand\figurename{\textbf{Figure}}
	%目录居中
	\renewcommand*\contentsname{\hfill Contents \hfill}
	% 修改参考文献标题
	\renewcommand{\refname}{Reference}
%

%页面参数
	%页边距
	% \geometry{top=0.8in,bottom=1in,left=0.5in,right=0.5in}
	%行距
	% \linespread{1.2}
	%目录深度
	% \setcounter{secnumdepth}{3}
%

%图片路径（实现多种文件夹都可以放图片
	% \graphicspath{{figures/}{figure/}{pictures/}{picture/}{pic/}{pics/}{image/}{images/}}
	\graphicspath{{pic/}}
%


\lstset{
	language=c++,
    basicstyle          =   \small\fontspec{Consolas},          % 基本代码风格
	morekeywords		=	{ll},	%增加高亮
    keywordstyle        =   \color{blue!90},          % 关键字风格
    commentstyle        =   \color{green!40!black!60},  % 注释的风格，斜体
    stringstyle         =   \ttfamily\slshape\color[RGB]{140,81,25},  % 字符串风格
    % framerule           =   2.5em,
	rulesepcolor		=	\color{red!20!green!20!blue!20},%代码块边框为淡青色
	framextopmargin=2pt,framexbottommargin=2pt,abovecaptionskip=-3pt,belowcaptionskip=3pt,%阴影边框设定
	rulecolor           =   \color{gray!10},  
	flexiblecolumns     =   flase,                % 别问为什么，加上这个
    numbers             =   left,   % 行号的位置在左边
    showspaces          =   false,  % 是否显示空格，显示了有点乱，所以不显示了
    numberstyle         =   \footnotesize\fontspec{Consolas}\emptyaccsupp,    % 行号的样式，小五号，tt等宽字体
    showstringspaces    =   false,
    captionpos          =   t,      % 这段代码的名字所呈现的位置，t指的是top上面
    frame               =   shadowbox,   % 显示边框
    tabsize             =   4,
    breaklines          =   true,   % 自动换行，建议不要写太长的行
    breakatwhitespace   =   false,       	% 设置自动中断是否只发生在空格处
    escapeinside        =   ``,     	% 在``里显示中文
    columns             =   fixed,  % 如果不加这一句，字间距就不固定，很丑，必须
    % basewidth           =    0.6em,
    xleftmargin         =    1em,  
    xrightmargin        =    1em,     % 设定listing左右的空白
}
