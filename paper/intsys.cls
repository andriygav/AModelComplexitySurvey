\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{intsys}[2026/02/19 Intelligent Systems journal class (amsbib, bilingual meta, fixed papersize)]
\newcommand*\DoNotLoadEpstopdf{} % do not load epstopdf package (suppresses warnings about shell escape)

\makeatletter
% --- options ---
\newif\ifintsyslang@ru
\intsyslang@rutrue

\newif\ifintsys@requireonlytitle
\intsys@requireonlytitlefalse

\newif\ifglobalthmnums
\globalthmnumsfalse

\newif\ifthmnumsshared
\thmnumssharedfalse

\RequirePackage{kvoptions}

\SetupKeyvalOptions{
  family=intsys,
  prefix=intsys@
}
\DeclareVoidOption{russian}{\intsyslang@rutrue}
\DeclareVoidOption{english}{\intsyslang@rufalse}
\DeclareVoidOption{ru}{\intsyslang@rutrue}
\DeclareVoidOption{en}{\intsyslang@rufalse}

\DeclareBoolOption{proceedings}
\DeclareBoolOption{editorial}

\DeclareStringOption[global]{thmwithin}
\DeclareStringOption[independent]{thmnums}

\ExecuteOptions{russian}
\ProcessKeyvalOptions*
\relax

\ifintsys@editorial
\intsys@requireonlytitletrue
\fi
\ifintsys@proceedings
\intsys@requireonlytitletrue
\fi

\def\intsys@thmnums@shared{\thmnumssharedtrue}
\def\intsys@thmnums@independent{\thmnumssharedfalse}

\ifcsname intsys@thmnums@\intsys@thmnums\endcsname
    \csname intsys@thmnums@\intsys@thmnums\endcsname
\else
    \ClassError{intsys}{Invalid thmnums value: \intsys@thmnums. Use 'shared' or 'independent'}{}%
\fi

\def\intsys@temp@global{global}
\ifx\intsys@thmwithin\intsys@temp@global%
    \globalthmnumstrue%
\else%
    \def\intsys@thmwithin@section{}
    \def\intsys@thmwithin@subsection{}
    \def\intsys@thmwithin@subsubsection{}
    \ifcsname intsys@thmwithin@\intsys@thmwithin \endcsname \else
    %\@ifundefined{c@\intsys@thmwithin}{%
      \ClassError{myclass}{Invalid thmwithin value: \intsys@thmwithin. Use 'global', 'section', 'subsection', or 'subsubsection'}{}%
    %}{}%
    \fi
\fi%

% --- end options ---

% --- external settings ---
% If you need to change something, do it in _intsys_settings.in
% Options:
\newif\ifintsysheaders % if true, enable headers/footers
\intsysheaderstrue  % default: no headers/footers
\newif\ifintsysexportmeta % if true, export meta to files
\intsysexportmetafalse  % default: no export
\newif\ifintsysprintacceptdate % if true, print "Received on ..."
\intsysprintacceptdatetrue  % default: print date
\newif\ifseparateauthorinfo % if true, print RU and EN author info separately
\separateauthorinfofalse  % default: no separate info
\newif\ifasymmetricmargins
\asymmetricmarginsfalse
\newif\ifprintversion
\printversionfalse
\newif\ifreviewversion
\reviewversionfalse
\newif\ifcommentpanel
\commentpanelfalse
\newif\ifintsysmicrotype % if true, load microtype package (requires scalable fonts)
\intsysmicrotypefalse  % default: disabled (may fail with old/incomplete TeX distributions)

%\reviewversiontrue

\IfFileExists{_intsys_settings.in}{\input{_intsys_settings.in}}{}

\newif\ifcheckmissingfields % if true, check for missing fields and issue errors
\checkmissingfieldstrue  % default: check for missing fields

% --- Load base class ---
\LoadClass[11pt,twoside]{article}

% --- fixed visual stack ---
\RequirePackage[T1,T2A]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{amsmath,amssymb,amsthm,mathtools}
\ifintsyslang@ru
    \RequirePackage[russian,english]{babel}
\else
    \RequirePackage[english]{babel}
\fi

\ifintsysmicrotype
  \RequirePackage{microtype}
\fi

%\RequirePackage{paratype}  % должен хорошо пропечатываться, но для математики неестественный
%\RequirePackage{cmsrb}  % неплохой шрифт, но буква "б" странная
%\RequirePackage{libertinus}  % неплохой шрифт, но надо чуть уменьшать межстрочные интервалы и отступы, чтобы выглядело пропорционально

\ifreviewversion
\RequirePackage[mathlines]{lineno}
\setlength\linenumbersep{6mm}
\renewcommand\thelinenumber{\textcolor{gray!70!white}{\arabic{linenumber}}}
\linenumbers
\fi

%\RequirePackage{caption}
\RequirePackage{fancyhdr}
\RequirePackage{geometry}
\RequirePackage{environ}
\RequirePackage{xcolor}
\RequirePackage[noadjust]{cite}

% --- базовые длины (вместо \def) ---
\newlength{\intsys@paperwidth}
\newlength{\intsys@paperheight}
\newlength{\intsys@innermargin}
\newlength{\intsys@outermargin}
\newlength{\intsys@topmargin}
\newlength{\intsys@bottommargin}
\newlength{\intsys@headertextwidth}

%\commentpaneltrue
% (если нужна панель комментариев справа)
\newlength{\sidewidth}
\ifcommentpanel
    \usepackage[textsize=footnotesize]{todonotes}

    \@mparswitchfalse   % выключить twoside-перекидку marginpar
    \normalmarginpar    % поместить на правую сторону
    \newcommand{\@IfNotEmpty}[2]{%
      \edef\@temp{#1}%
      \ifx\@temp\@empty
      \else
        #2%
      \fi
    }
    \newenvironment{DIFnomarkup}{}{}
    \newcommand{\sidecomment}[3]{%
    \foreignlanguage{russian}{\todo[backgroundcolor=yellow!20,linecolor=red,tickmarkheight=1mm]{\@IfNotEmpty{#1#2}{\textbf{#1}\@IfNotEmpty{#2}{ (#2)}:\\[2pt]\hrule\vspace{3pt}}
      #3}}%
    }
    \newcommand{\scommentselection}{\color{red}}
    \newcommand{\ecommentselection}{\color{black}}
    \ifprintversion
        \ifasymmetricmargins
            \ClassError{intsys}{Options printversion and commentpanel are incompatible.}{When comments panel is used, \string\printversionfalse should be set}
            \printversionfalse
        \fi
    \fi
    \setlength{\sidewidth}{50mm}
    \setlength{\marginparwidth}{50mm}
    \setlength{\marginparsep}{10mm}
\else
    \setlength{\sidewidth}{0mm}
\fi

% --- значения по умолчанию ---
\setlength{\intsys@paperwidth}{165mm}
\setlength{\intsys@paperheight}{235mm}
\setlength{\intsys@headertextwidth}{115mm}

\ifasymmetricmargins
  \ifprintversion
    \setlength{\intsys@innermargin}{23mm}
  \else
    \setlength{\intsys@innermargin}{13mm}
    \setlength{\intsys@paperwidth}{155mm}
  \fi
  \setlength{\intsys@outermargin}{13mm}
  \setlength{\intsys@topmargin}{17mm}
  \setlength{\intsys@bottommargin}{15mm}
\else
  \setlength{\intsys@innermargin}{19mm}
  \setlength{\intsys@outermargin}{19mm}
  \ifintsysheaders
    \setlength{\intsys@topmargin}{20mm}
    \setlength{\intsys@bottommargin}{18mm}
  \else
    \setlength{\intsys@topmargin}{18mm}
    \setlength{\intsys@bottommargin}{21mm}
  \fi
\fi


% --- geometry ---
\geometry{
    paperwidth=\dimexpr\intsys@paperwidth\relax,
    paperheight=\intsys@paperheight,
    inner=\intsys@innermargin,
    outer=\intsys@outermargin,
    % \fi
    top=\intsys@topmargin,
    bottom=\intsys@bottommargin,
    marginparwidth=\sidewidth,
    headheight=26pt, headsep=10pt,
    footskip=20pt,
    heightrounded
}
\ifcommentpanel
    \addtolength{\paperwidth}{\dimexpr\sidewidth\relax}
    \addtolength{\pdfpagewidth}{\dimexpr\sidewidth\relax} %
\fi


% --- macros for authors namesformatting ---

\RequirePackage{listofitems}
\RequirePackage{trimspaces}

\DeclareRobustCommand*\firstletter[1]{%
  \begingroup
    \edef\first@tmp{#1}%
    \expandafter\first@dispatch\first@tmp\first@stop
  \endgroup
}

% #1 = первый токен, #2 = хвост до \first@stop
\def\first@dispatch#1#2\first@stop{%
  % Если первый токен — control sequence, это не "байт"
  \ifcat\noexpand#1\relax
    % Старый inputenc часто даёт \IeC{...}
    \ifx#1\IeC
      \expandafter\first@from@IeC#2\first@stop
    \else
      % Частый кириллический случай: \T2A\CYRA -> вывести два токена
      \ifx#1\T2A #1#2\else
      \ifx#1\T2B #1#2\else
      \ifx#1\T2C #1#2\else
        % Иначе считаем, что первый токен сам уже "буква"
        #1%
      \fi\fi\fi
    \fi
  \else
    % Иначе первый токен — символ/активный байт: можно разбирать UTF-8
    \def\first@pick##1##2\first@stop{%
      \ifnum`##1<"80 %
        \first@emit{##1}%
      \else\ifnum`##1<"E0 %
        \first@two##1##2\first@stop
      \else\ifnum`##1<"F0 %
        \first@three##1##2\first@stop
      \else
        \first@four##1##2\first@stop
      \fi\fi\fi
    }%
    \def\first@two##1##2##3\first@stop{\first@emit{##1##2}}%
    \def\first@three##1##2##3##4\first@stop{\first@emit{##1##2##3}}%
    \def\first@four##1##2##3##4##5\first@stop{\first@emit{##1##2##3##4}}%
    % Пересканировать байты в символ, но НИ ОДНОГО лишнего токена в хвост
    \def\first@emit##1{%
      \begingroup
        \endlinechar=-1 %
        \everyeof{}%
        \scantokens{##1}%
      \endgroup
    }%
    \first@pick #1#2\first@stop
  \fi
}

% Основной макрос для форматирования ФИО
\newcommand{\formatFIOs}[1]{%
  % Разбиваем по пробелам
  \setsepchar{ }%
  \readlist*\words{#1}%
  % Проверяем есть ли имя
  \ifnum\wordslen>1%
    % Обрабатываем имя
    \processInitial{\words[2]}%
  \fi%
  % Проверяем есть ли отчество
  \ifnum\wordslen>2%
    % Обрабатываем отчество
    \,\processInitial{\words[3]}%
  \fi%
  \ifnum\wordslen>1%
    ~%
  \fi%
  % Выводим фамилию
  \words[1]%
}
\newcommand{\formatFIO}[1]{
    %\message{formatFIO: #1}
    \edef\trimmed{\trim@spaces{#1}}%
    \expandafter\formatFIOs{\trimmed}%
}

% Макрос для обработки инициала (с поддержкой дефисов)
\newcommand{\processInitial}[1]{%
  % Проверяем наличие дефиса
  \setsepchar{-}%
  \readlist*\parts{#1}%
  \ifnum\partslen=1
    % Нет дефиса - просто первая буква
    \firstletter{#1}.%
  \else
    % Есть дефис - обрабатываем части
    \foreachitem\part\in\parts{%
      \ifnum\partcnt=1
        \firstletter{\part}.%
      \else
        -\firstletter{\part}.%
      \fi
    }%
  \fi
}

% --- headers/footers ---
% --- bilingual metadata (без '@' в именах) ---
\newcommand\IntsysKeywordsRU{\@empty}
\newcommand\IntsysKeywordsEN{\@empty}
\let\IntsysTitleRU\@empty
\let\IntsysTitleEN\@empty
\let\IntsysAbstractRU\@empty
\let\IntsysAbstractEN\@empty
\let\IntsysAuthorsRU\@empty
\let\IntsysAuthorsEN\@empty
\let\IntsysAuthorsList\@empty % NameRU|AffRU|NameEN|AffEN|email|id ;;; ...
\let\IntsysBibliographyEN\@empty
\let\intsys@AuthorsRUDisplay\@empty
\let\intsys@AuthorsENDisplay\@empty
\let\IntsysReceivedDateRU\@empty
\let\IntsysReceivedDateEN\@empty

\newcommand{\ReceivedDate}[2]{%
\gdef\IntsysReceivedDateRU{#1}%
\gdef\IntsysReceivedDateEN{#2}%
}
% import dates from ___received_meta.tex if exists
\IfFileExists{_received_date.in}{\input{_received_date.in}}{}

\newcommand\TitleRU[1]{
    \ifx\IntsysTitleRU\@empty
        \gdef\IntsysTitleRU{#1}
    \else
        \ClassError{intsys}{TitleRU is already set}{You can set the title in Russian only once using \string\TitleRU\{...\}}
    \fi
}
\newcommand\TitleEN[1]{
    \ifx\IntsysTitleEN\@empty
        \gdef\IntsysTitleEN{#1}
    \else
        \ClassError{intsys}{TitleEN is already set}{You can set the title in English only once using \string\TitleEN\{...\}}
    \fi
}
\newcommand\AbstractRU[1]{
    \ifx\IntsysAbstractRU\@empty
        \gdef\IntsysAbstractRU{#1}
    \else
        \ClassError{intsys}{AbstractRU is already set: """\IntsysAbstractRU""". Trying to redefine it with: """#1"""}{You can set the abstract in Russian only once using \string\AbstractRU\{...\}}
    \fi
}
\newcommand\AbstractEN[1]{
    \ifx\IntsysAbstractEN\@empty
        \gdef\IntsysAbstractEN{#1}
    \else
        \ClassError{intsys}{AbstractEN is already set}{You can set the abstract in English only once using \string\AbstractEN\{...\}}
    \fi
}
\newcommand\KeywordsRU[1]{\gdef\IntsysKeywordsRU{#1}}
\newcommand\KeywordsEN[1]{\gdef\IntsysKeywordsEN{#1}}

\let\IntShortTitle\@empty
\let\IntShortAuthor\@empty
\newcommand\ShortTitle[1]{\gdef\IntShortTitle{#1}}
\newcommand\IntSetShortAuthor[1]{\gdef\IntShortAuthor{#1}}

\newcommand{\authorsymbol}[1]{%
    \ifcase\value{#1}
    \or \ensuremath{*}%
    \or \,\ddag%
    \or \S%
    \or \P%
    \or \textbardbl%
    \or \ensuremath{**}%
    \or \,\ddag\ddag%
    \or \S\S%
    \or \dag\dag%
    \or \P\P%
    \else
       \roman{#1}%
    \fi
}

%\newcounter{authorinfo}
\newcommand{\authorinfo}[1]{%
    \begingroup%
    \renewcommand{\thefootnote}{\authorsymbol{footnote}}%
    \footnote{#1}
    %\footnotetext[\fnsymbol{authorinfo}]{#1}%
    \endgroup%
}

\newcommand{\intsys@ifnotempty}[2]{\if\relax\detokenize{#1}\relax\else #2\fi}
\newcommand{\intsys@formatAuthorNote}[6]{%
  \authorinfo{%
  \ifintsyslang@ru
      \textit{\trim@spaces{#1}} %\intsys@ifnotempty{#6}{\,\orcidlink{#6}}
       --- \trim@spaces{#2}\intsys@ifnotempty{#5}{, e-mail: \url{\trim@spaces{#5}}}%
       \intsys@ifnotempty{#6}{, ORCID: #6}%
  \fi
  \ifseparateauthorinfo\else%
    .\\\indent \emph{\trim@spaces{#3}} --- \trim@spaces{#4}.%
  \fi}%
}
\newcommand{\intsys@formatAuthorNoteEN}[6]{%
  \ifintsyslang@ru
  \ifseparateauthorinfo%
    \authorinfo{\textit{\trim@spaces{#3}} --- \trim@spaces{#4}\intsys@ifnotempty{#5}{, e-mail: \url{\trim@spaces{#5}}}}\intsys@ifnotempty{#6}{, ORCID: #6}%
  \fi%
  \else
    \authorinfo{\textit{\trim@spaces{#3}} --- \trim@spaces{#4}\intsys@ifnotempty{#5}{, e-mail: \url{\trim@spaces{#5}}}}\intsys@ifnotempty{#6}{, ORCID: #6}%
  \fi
}

\def\intsys@prepareAuthors{% build authors with footnotes
  \global\let\intsys@AuthorsRUDisplay\@empty
  \global\let\intsys@AuthorsENDisplay\@empty
  \ifx\IntsysAuthorsList\@empty\else
        \expandafter\intsys@BuildAuthorsMacro\IntsysAuthorsList;;;\IntsysEnd
        \setcounter{footnote}{0}
  \fi
}
\def\intsys@BuildAuthorsMacro#1;;;#2\IntsysEnd{%
  \intsys@AddAuthorMacro#1\relax
  \ifx\relax#2\relax\else
    \intsys@BuildAuthorsMacro#2\IntsysEnd
  \fi
}
\def\intsys@AddAuthorMacro#1|#2|#3|#4|#5|#6\relax{%
  \ifx\intsys@AuthorsRUDisplay\@empty
    \ifintsyslang@ru
        \gdef\intsys@AuthorsRUDisplay{\formatFIO{#1}\intsys@formatAuthorNote{#1}{#2}{#3}{#4}{#5}{#6}}%
    \fi
    \gdef\intsys@AuthorsENDisplay{\formatFIO{#3}\intsys@formatAuthorNoteEN{#1}{#2}{#3}{#4}{#5}{#6}}%
  \else
    \ifintsyslang@ru
        \g@addto@macro\intsys@AuthorsRUDisplay{, \formatFIO{#1}\intsys@formatAuthorNote{#1}{#2}{#3}{#4}{#5}{#6}}%
    \fi
    \g@addto@macro\intsys@AuthorsENDisplay{, \formatFIO{#3}\intsys@formatAuthorNoteEN{#1}{#2}{#3}{#4}{#5}{#6}}%
  \fi
}

\ifintsyslang@ru
% ---- Макрос для авторов в русской версии
% \Author{NameRU}{AffRU}{NameEN}{AffEN}{email}{id}
\newcommand\Author[6]{%
  \ifx\IntsysAuthorsRU\@empty
    \gdef\IntsysAuthorsRU{\formatFIO{#1}}%
    \gdef\IntsysAuthorsEN{\formatFIO{#3}}%
  \else
    \g@addto@macro\IntsysAuthorsRU{, \formatFIO{#1}}%
    \g@addto@macro\IntsysAuthorsEN{, \formatFIO{#3}}%
  \fi
  \ifx\IntsysAuthorsList\@empty
    \gdef\IntsysAuthorsList{#1|#2|#3|#4|#5|#6}%
  \else
    \g@addto@macro\IntsysAuthorsList{;;;#1|#2|#3|#4|#5|#6}%
  \fi
  \ifx\IntShortAuthor\@empty \IntSetShortAuthor{\formatFIO{#1}}
  \else
    \g@addto@macro\IntShortAuthor{, \formatFIO{#1}}%
  \fi
}
\else
% ---- Макрос для авторов в английской версии
% \Author{NameEN}{AffEN}{email}{id}
\newcommand\Author[6]{%
  \ifx\IntsysAuthorsEN\@empty
    \gdef\IntsysAuthorsRU{\formatFIO{#3}}%
    \gdef\IntsysAuthorsEN{\formatFIO{#3}}%
  \else
    \g@addto@macro\IntsysAuthorsRU{, \formatFIO{#3}}%
    \g@addto@macro\IntsysAuthorsEN{, \formatFIO{#3}}%
  \fi
  \ifx\IntsysAuthorsList\@empty
    \gdef\IntsysAuthorsList{||#3|#4|#5|#6}%
  \else
    \g@addto@macro\IntsysAuthorsList{;;;||#3|#4|#5|#6}%
  \fi
  \ifx\IntShortAuthor\@empty \IntSetShortAuthor{\formatFIO{#3}}
  \else
     \g@addto@macro\IntShortAuthor{, \formatFIO{#3}}%
  \fi
}
\fi

\ifintsys@proceedings
    \RequirePackage{needspace}
    \newcommand{\EventDate}[1]{\needspace{3.5cm}
    \begin{center}#1\end{center}}
    \newcommand{\EventTitle}[2]{\begin{center}

        {\Large \bf #1}

        \medskip

        #2
    \end{center}}
\fi
% \AtBeginDocument{%
%   \ifintsyslang@ru
%     \selectlanguage{russian}% ensure russian hyphenation tables are active
%   \else
%     \selectlanguage{english}% fallback for english-only articles
%   \fi
% }
% running heads
\ifintsysheaders

\pagestyle{fancy}
\fancyhf{}

\newcommand{\CheckRunningWidth}[4]{%
  % #1 — имя (для сообщения)
  % #2 — допустимая ширина
  % #3 — содержимое
  \begingroup
    \setbox0=\hbox{#3}%
    \ifdim\wd0>#2
      \newcommand{\DimToMM}[1]{\fpeval{round((##1)/1mm,2)}}
      \ClassError{intsys}{%
        #1 is too wide in running header; #4\MessageBreak
        Width = \DimToMM{\wd0} mm\MessageBreak
        Limit = \DimToMM{#2} mm%
      }{%
        #4.%
      }%
    \fi
  \endgroup
}


\newcommand\intsysApplyRunning{%
  \CheckRunningWidth{Title}{\intsys@headertextwidth}{\small\IntShortTitle}{Set short title using \string\ShortTitle\{...\}}
  \CheckRunningWidth{Authors}{\intsys@headertextwidth}{\small\IntShortAuthor}{Set shortened authors list \string\IntSetShortAuthor\{...\}}
  \ifintsyslang@ru
    \fancyhead[CE]{\foreignlanguage{russian}{\small\IntShortAuthor}}%
    \fancyhead[CO]{\foreignlanguage{russian}{\small\IntShortTitle}}%
    \fancypagestyle{firstpage}{
        \fancyhf{}%
        \fancyhead{}%
        \renewcommand{\headrulewidth}{0pt}%
        \fancyfoot[R]{\scriptsize\foreignlanguage{russian}{Опубликовано на условиях лицензии CC-BY 4.0\\[2pt]
        Published under CC-BY 4.0}}%
    }
    %\fancypagestyle{entitlepage}{
    %    \fancyfoot[R]{\scriptsize\foreignlanguage{english}{Published under CC-BY 4.0}}%
    %}
  \else
    \fancyhead[CE]{\foreignlanguage{english}{\small\IntShortAuthor}}%
    \fancyhead[CO]{\foreignlanguage{english}{\small\IntShortTitle}}%
    \fancypagestyle{firstpage}{
        \fancyhf{}
        \fancyfoot[R]{\scriptsize\foreignlanguage{english}{Published under CC-BY 4.0}}%
    }
  \fi
  \fancyhead[LE]{\thepage}\fancyhead[RO]{\thepage}%
}
\AtBeginDocument{\intsysApplyRunning}
\fi

% default short titles = full titles
\AtBeginDocument{%
  \ifintsyslang@ru
    \selectlanguage{russian}% ensure russian hyphenation tables are active
  \else
    \global\let\IntsysTitleRU\IntsysTitleEN
    \global\let\IntsysAbstractRU\IntsysAbstractEN
    \global\let\IntsysKeywordsRU\IntsysKeywordsEN
    \selectlanguage{english}% english-only articles
  \fi
  \ifx\IntShortTitle\@empty \expandafter\ShortTitle\expandafter{\IntsysTitleRU}\fi
  \IfFileExists{_init_document.tex}{\input{_init_document}}{}
  \ifcheckmissingfields
    \ifintsyslang@ru
      \ifx\IntsysTitleRU\@empty
        \ClassError{intsys}{TitleRU is not set}{You must set thetitle in Russian using \string\TitleRU\{...\}}
      \fi
      \ifintsys@requireonlytitle\else
        \ifx\IntsysAbstractRU\@empty
            \ClassError{intsys}{AbstractRU is not set}{You must setthe abstract in Russian using \string\AbstractRU\{...\}}
        \fi
        \ifx\IntsysKeywordsRU\@empty
            \ClassError{intsys}{KeywordsRU is not set}{You must setthe keywords in Russian using \string\KeywordsRU\{...\}}
        \fi
      \fi
    \fi
    \ifintsys@requireonlytitle\else
        \ifx\IntsysTitleEN\@empty
            \ClassError{intsys}{TitleEN is not set}{You must set the title in English using \string\TitleEN\{...\}}
        \fi
        \ifx\IntsysAbstractEN\@empty
            \ClassError{intsys}{AbstractEN is not set}{You must set the abstract in English using \string\AbstractEN\{...\}}
        \fi
        \ifx\IntsysKeywordsEN\@empty
            \ClassError{intsys}{KeywordsEN is not set}{You must set the keywords in English using \string\KeywordsEN\{...\}}
        \fi
        \ifx\IntsysAuthorsEN\@empty
            \ClassError{intsys}{No authors defined}{You must define at least one author using \string\Author\{...\}\{...\} ...}
        \fi
    \fi
  \fi
  \unitlength=1mm
  \maketitle
  \renewcommand{\maketitle}{
    \ClassWarning{intsys}{\string\maketitle ignored: it shouldn't be called manually}{}
  }
}


% --- title block ---
\renewcommand\maketitle{%
  \setcounter{footnote}{0} % TODO: separate counter for authors
  \ifintsysheaders%
    \thispagestyle{firstpage}%
  \fi%
  \intsys@prepareAuthors
  \ifintsys@proceedings
    \begin{center}
      \LARGE \IntsysTitleRU
    \end{center}

    \IntsysAbstractRU

    \medskip
    \bigskip
  \else\ifintsys@editorial
    \begin{center}
      \LARGE \IntsysTitleRU
    \end{center}

    \IntsysAbstractRU

    \medskip
    \bigskip

  \else
    \begin{center}
      {\begin{spacing}{1.0}\LARGE \bf %
      \IntsysTitleRU
      \end{spacing}
      }
      \hfill
      \break
      \large
      %\vspace{1.0em}
      \ifintsyslang@ru
        {\intsys@AuthorsRUDisplay}
      \else
        {\intsys@AuthorsENDisplay}
      \fi
    \end{center}
    %\vspace{0.0em}
    \begin{quotation}
        \small
        \IntsysAbstractRU\par
        \ifx\IntsysKeywordsRU\@empty\else
        \smallskip
        \ifintsyslang@ru%
            \textbf{Ключевые слова:}%
        \else%
            \textbf{Keywords:}%
        \fi%
        \ \IntsysKeywordsRU.\par
        \fi
    \end{quotation}
  \fi\fi
  \setcounter{footnote}{0}

  % turn on line numbers (version for reviewers)
  \ifreviewversion
    \linenumbers
  \fi
  %\vspace{0.0em}
}

% --- bibliographies ---
\newcommand{\@bibstyle}{%
    \baselineskip3.8mm
    \parskip-0.5pt
    \def\newblock{\hskip .11em plus .33em minus .07em}%
    \sloppy\clubpenalty4000\widowpenalty4000
    \sfcode`\.=1000\relax
    \def\makelabel##1{\hfil ##1}%
}
\newcommand\intsys@startbibliographyRU[1]{%
  \begingroup
    \selectlanguage{russian}%
    \def\refname{Список литературы}%
    \def\bibname{Список литературы}%
    \refstepcounter{aux@counter}%
    \addcontentsline{toc}{section}{\refname}%
    \section*{\refname}%
    \def\@biblabel##1{{\select@language{russian}[##1]}}%
    \small
    \list{\@biblabel{\arabic{enumi}}}{\settowidth\labelwidth{\@biblabel{#1}}\leftmargin\labelwidth
      \advance\leftmargin\labelsep
      \usecounter{enumi}}%
    %\@bibstyle{}%
}
\newcommand\intsys@startbibliographyEN[1]{%
  \begingroup
    \ifintsyslang@ru
        \let\oldBibitem\Bibitem
        \let\oldbibitem\bibitem
        \renewcommand{\Bibitem}[1]{\oldBibitem{##1:en-refs}}
        \renewcommand{\RBibitem}[1]{\oldBibitem{##1:en-refs}}
        \renewcommand{\bibitem}[1]{\oldbibitem{##1:en-refs}}
    \fi
    \selectlanguage{english}%
    \def\refname{References}%
    \def\bibname{References}%
    \refstepcounter{aux@counter}%
    \addcontentsline{toc}{section}{\refname}%
    \section*{\refname}%
    \def\@biblabel##1{{\select@language{english}[##1]}}%
    \small
    \list{\@biblabel{\arabic{enumi}}}{\settowidth\labelwidth{\@biblabel{#1}}\leftmargin\labelwidth
      \advance\leftmargin\labelsep
      \usecounter{enumi}}%
    %\@bibstyle{}%
}
\newcommand\intsys@endbibliography{%
    \endlist
  \endgroup
}
\newenvironment{thebibliographyRU}[1]{\thebibliography{#1}}{
    
\endthebibliography}
    %\begin{thebibliography}{#1}}{\end{thebibliography}}
%{intsys@startbibliographyRU{#1}}{\intsys@endbibliography}

\let\IntsysBibliographyENbody\@empty
\NewEnviron{thebibliographyEN}[1]{%
  \global\let\IntsysBibliographyENbody\BODY
  \global\def\IntsysBibliographyEN{\intsys@startbibliographyEN{#1}\IntsysBibliographyENbody\intsys@endbibliography}%
}

% --- meta dump (без '@', без групп в хуках) ---
\newwrite\intsystmpw
\newcommand{\IntsysWriteFile}[2]{%
  \immediate\openout\intsystmpw=#1\relax
  \immediate\write\intsystmpw{\detokenize\expandafter{#2}}%
  \immediate\closeout\intsystmpw
}

\protected\def\IntsysDumpMeta{%
  \ifintsys@proceedings
    \IntsysWriteFile{\jobname.type.txt}{proceedings}%
  \else\ifintsys@editorial
    \IntsysWriteFile{\jobname.type.txt}{editorial}%
  \else
    \IntsysWriteFile{\jobname.type.txt}{regular}%
  \fi\fi
  \ifintsyslang@ru
  \IntsysWriteFile{\jobname.title.ru.tex}{\IntsysTitleRU}%
  \IntsysWriteFile{\jobname.abstract.ru.tex}{\IntsysAbstractRU}%
  \IntsysWriteFile{\jobname.keywords.ru.txt}{\IntsysKeywordsRU}%
  \else
  \IntsysWriteFile{\jobname.title.ru.tex}{}%
  \IntsysWriteFile{\jobname.abstract.ru.tex}{}%
  \IntsysWriteFile{\jobname.keywords.ru.txt}{}%
  \fi
  \IntsysWriteFile{\jobname.title.en.tex}{\IntsysTitleEN}%
  \IntsysWriteFile{\jobname.abstract.en.tex}{\IntsysAbstractEN}%
  \IntsysWriteFile{\jobname.keywords.en.txt}{\IntsysKeywordsEN}%
  \IntsysWriteFile{\jobname.authors.tsv}{\IntsysAuthorsList}%
}
\protected\def\IntsysPrintEnglishTail{%
  %\clearpage
  \begingroup
    \setcounter{footnote}{0} % TODO: separate counter for authors
    \vspace{5mm}
    \selectlanguage{english}%
    \normalsize
    \ifx \IntsysTitleEN\@empty
      \gdef\IntsysTitleEN{\color{red} TODO: provide english title using \textbackslash{}TitleEN\{...\} macro}
    \fi
    \ifx \IntsysAbstractEN\@empty
      \gdef\IntsysAbstractEN{\color{red} TODO: provide english abstract using \textbackslash{}AbstractEN\{...\} macro}
    \fi
    \begin{center}
      {\bf \IntsysTitleEN}\par%
      \medskip%
      {\intsys@AuthorsENDisplay \par}
    %\vspace{0.0em}%
    \begin{quotation}
      \small
      \IntsysAbstractEN\par
      \ifx\IntsysKeywordsEN\@empty\else
        \smallskip
        \textbf{Keywords:}\ \IntsysKeywordsEN.\par
      \fi
    \end{quotation}
    \end{center}%\thispagestyle{entitlepage}
    \ifx\IntsysBibliographyEN\@empty\else\IntsysBibliographyEN\fi
  \endgroup
}
\AtEndDocument{%
    \ifintsys@requireonlytitle\else
        \ifintsysprintacceptdate
            \ifintsyslang@ru
                \ifx\IntsysReceivedDateRU\@empty\else
                    \nopagebreak
                    %\smallskip
                    %\begin{flushright}
                    {\hfill \small Статья поступила \IntsysReceivedDateRU}
                    %\end{flushright}
                \fi
            \else
                \ifx\IntsysReceivedDateEN\@empty\else
                    %\smallskip
                    \nopagebreak
                    {\hfill \small  Received on \IntsysReceivedDateEN}
                \fi
            \fi
        \fi
        \ifintsyslang@ru%
            \smallskip
            \IntsysPrintEnglishTail%
            \ifintsysprintacceptdate
                \ifx\IntsysReceivedDateEN\@empty\else
                    \nopagebreak
                    %\smallskip
                    {\hfill \small Received on \IntsysReceivedDateEN}
                \fi
            \fi
        \fi
    \fi
    \ifintsysexportmeta
        \IntsysDumpMeta
    \fi
}
% --- mathematical environments ---

\ifintsyslang@ru
\def\intsys@theoremname{Теорема}
\else
\def\intsys@theoremname{Theorem}
\fi

\ifthmnumsshared
    \ifglobalthmnums
        \newtheorem{theorem}{\intsys@theoremname}
    \else
        \newtheorem{theorem}{\intsys@theoremname}[\intsys@thmwithin]
    \fi
    \newcommand{\intsys@newtheorem}[2]{\newtheorem{#1}[theorem]{#2}}
\else
    \ifglobalthmnums
        \newcommand{\intsys@newtheorem}[2]{\newtheorem{#1}{#2}}
    \else
        \newcommand{\intsys@newtheorem}[2]{\newtheorem{#1}{#2}[\intsys@thmwithin]}
    \fi
    \intsys@newtheorem{theorem}{\intsys@theoremname}
\fi

\ifintsyslang@ru % in Russian
  \theoremstyle{definition}
  \newtheorem*{definition*}{Определение}
  \intsys@newtheorem{definition}{Определение}
  \newtheorem*{example*}{Пример}
  \intsys@newtheorem{example}{Пример}

  \theoremstyle{remark}
  \newtheorem*{remark*}{Замечание}
  \intsys@newtheorem{remark}{Замечание}

  \theoremstyle{plain}

  %\intsys@newtheorem{theorem}{Теорема}
  \intsys@newtheorem{lemma}{Лемма}
  \intsys@newtheorem{corollary}{Следствие}
  \intsys@newtheorem{proposition}{Утверждение}

  \newtheorem*{hypothesis*}{Гипотеза}
  \intsys@newtheorem{hypothesis}{Гипотеза}

\else % in English
  \theoremstyle{definition}
  \newtheorem*{definition*}{Definition}
  \intsys@newtheorem{definition}{Definition}
  \newtheorem*{example*}{Example}
  \intsys@newtheorem{example}{Example}

  \theoremstyle{remark}
  \intsys@newtheorem{remark}{Remark}
  \newtheorem*{remark*}{Remark}

  \theoremstyle{plain}
  %\intsys@newtheorem{theorem}{Theorem}
  \intsys@newtheorem{lemma}{Lemma}
  \intsys@newtheorem{corollary}{Corollary}
  \intsys@newtheorem{proposition}{Proposition}

  \intsys@newtheorem{hypothesis}{Hypothesis}
  \newtheorem*{hypothesis*}{Hypothesis}
\fi
\makeatother

% --- appendix sections in Russian ---
\RequirePackage{titlesec}

\makeatletter

\let\oldappendix\appendix
\ifintsyslang@ru
\renewcommand{\appendix}{%
  \oldappendix
  \renewcommand{\thesection}{\Asbuk{section}}%
  \titleformat{\section}
    {\large\bfseries}
    {Приложение~\thesection.}
    {0.5em}
    {}
}
\else
\renewcommand{\appendix}{%
  \oldappendix
  \titleformat{\section}
    {\large\bfseries}
    {Appendix~\thesection.}
    {0.5em}
    {}
}
\fi

% section title format
\renewcommand{\@seccntformat}[1]{%
  \csname the#1\endcsname.\
}
% ------------------------------------

% --- typography tweaks ---
\tolerance=9999
\clubpenalty=10000
\widowpenalty=10000


% --- adjusting spacing settings ---
\RequirePackage{caption}
\captionsetup[table]{skip=5pt}

\RequirePackage{setspace}
\setlength{\itemsep}{0pt plus 2pt}        % между элементами
\setlength{\parsep}{0pt plus 2pt}
\RequirePackage{enumitem}
\setlist[itemize,enumerate]{%
  %leftmargin=2em,
  %labelsep=0.5em,
  itemsep=0.5ex plus 2pt,   % decrease item separation
  parsep=0.25ex plus 2pt,   % decrease paragraph separation in enumerate/itemize lists
  %topsep=0.3em
}

%\setlist[itemize]{%
%    leftmargin=1em}
\def\@textbottom{\vskip 0pt plus 20pt}

\renewcommand{\@listii}{%
  \setlength{\leftmargin}{1.4em}%
  \setlength{\topsep}{0.5ex plus 2pt}
  \setlength{\parsep}{0.2ex plus 2pt}%
  \setlength{\itemsep}{0.2ex plus 2pt}%
}
% \titlespacing*{\section}{0pt}{3.5ex plus 0.3ex minus 0.8ex}{2.3ex plus 0.2ex minus 0.5ex}
% \titlespacing*{\subsection}{0pt}{3.25ex plus 0.2ex minus 0.7ex}{1.5ex plus 0.2ex minus 0.3ex}
% \titlespacing*{\subsubsection}{0pt}{3.25ex plus 0.2ex minus 0.7ex}{1.5ex plus 0.2ex minus 0.3ex}
\titlespacing*{\section}{0pt}{3.5ex plus 0.3ex minus 0.2 ex}{2.3ex plus 0.2ex minus 0.2 ex}
\titlespacing*{\subsection}{0pt}{3.25ex plus 0.2ex minus 0.2 ex}{1.5ex plus 0.2ex minus 0.2 ex}
\titlespacing*{\subsubsection}{0pt}{3.25ex plus 0.2ex minus 0.2 ex}{1.5ex plus 0.2ex minus 0.2 ex}

% \RequirePackage{etoolbox}
% \AtBeginEnvironment{verbatim}{%
%   \setlength{\topsep}{1ex}%
%   \setlength{\partopsep}{1ex}%
% }
\RequirePackage{array}
\setlength{\extrarowheight}{2pt}  % extra row height for tables
% ---     end of spacing         ---

% ---   counters in Russian   ---
\ifintsyslang@ru
\renewcommand{\labelenumii}{\asbuk{enumii})}
\fi

\makeatletter
\AddEnumerateCounter{\asbuk}{\russian@alph}{я}
\makeatother

\setlist[enumerate,2]{label=\asbuk*), ref=\asbuk*)}
% --- end counters in Russian ---

% --- amsbib + hyperref (без конфликтов) ---
\PassOptionsToPackage{unicode}{hyperref}
\RequirePackage{amsbib}

\makeatletter
\@ifpackageloaded{hyperref}{}{%
  \RequirePackage{hyperref}%
}
%\RequirePackage{orcidlink}
\hypersetup{hidelinks}
\makeatother
% ------------------------------------------

% --- end of intsys.cls ---
