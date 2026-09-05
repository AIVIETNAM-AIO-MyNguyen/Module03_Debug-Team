\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ai_conquer2026}[2026/02/01 AI Conquer 2026 Class]

\LoadClass[12pt]{article}

\RequirePackage[utf8]{inputenc}
\RequirePackage[vietnamese]{babel}
\RequirePackage[margin=2.5cm]{geometry}
\RequirePackage{amsmath,amssymb,amsfonts}
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{hyperref}
\RequirePackage{xcolor}
\RequirePackage{titlesec}
\RequirePackage{titling}
\RequirePackage{enumitem}
\RequirePackage{listings}
\RequirePackage{tcolorbox}
\tcbuselibrary{skins,breakable,listings}

% Biblatex setup
\RequirePackage[
  backend=biber,
  style=numeric-comp,
  sorting=none
]{biblatex}

\definecolor{aivnGreen}{RGB}{0, 130, 60}
\definecolor{captionbg}{RGB}{240, 245, 240}

\hypersetup{
    colorlinks=true,
    linkcolor=aivnGreen,
    citecolor=blue,
    urlcolor=aivnGreen
}

\newtcolorbox{notebox}[2][]{enhanced,breakable,
  colback=gray!10!white,
  colframe=aivnGreen,
  fonttitle=\bfseries,
  title={Chú thích: #2},
  boxrule=0.8pt, arc=3pt,
  left=2mm, right=2mm, top=2mm, bottom=2mm,
  #1}

\newtcblisting{aivnoutputbox}[1][]{%
  enhanced, breakable,
  listing engine=listings, listing only,
  colback=white, colframe=aivnGreen, coltext=black!85,
  coltitle=white, title={#1}, fonttitle=\bfseries\sffamily,
  left=6pt,right=6pt,top=6pt,bottom=6pt, arc=2mm, boxrule=0.4pt,
  listing options={basicstyle=\ttfamily\footnotesize, breaklines=true, numbers=none}
}

\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
