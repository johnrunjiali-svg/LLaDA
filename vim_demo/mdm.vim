" Companion for vim_demo/mdm_vim.py:   vim -S vim_demo/mdm.vim mdm.txt
scriptencoding utf-8

" The watcher rewrites the file after every :w. Reload it whenever there are no unsaved edits.
setlocal autoread noswapfile nowrap nolist nospell nofoldenable cursorline cursorcolumn
if exists('g:mdm_timer') | call timer_stop(g:mdm_timer) | endif
let g:mdm_timer = timer_start(250, {-> mode() ==# 'n' && !&modified ? execute('silent! checktime') : 0}, {'repeat': -1})

" :Mdm auto 2   puts a command on the CMD line and saves, without moving the cursor
command! -buffer -nargs=* Mdm call setline(search('^CMD\s', 'nw'), 'CMD    ✎ > ' . <q-args>) | write

if !exists('g:syntax_on') | syntax enable | endif
syntax clear
syntax match mdmSep    /[│┼┤─]/
syntax match mdmBar    /[▏▎▍▌▋▊▉█]\+/
syntax match mdmMask   /\[M\]/
syntax match mdmDone   /✔[^│]*/
syntax match mdmChosen /▸[^│]*/
syntax match mdmEdit   /^\S\+\s*✎/
syntax match mdmError  /ERROR.*$/
syntax match mdmHelp   /^# .*$/
highlight default link mdmSep    NonText
highlight default link mdmBar    Type
highlight default link mdmMask   Special
highlight default link mdmDone   String
highlight default link mdmChosen Search
highlight default link mdmEdit   Statement
highlight default link mdmError  Error
highlight default link mdmHelp   Comment
