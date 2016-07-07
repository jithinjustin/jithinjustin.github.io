---
layout: post
title: "Blogging with Jekyll"
date: "2016-01-06 14:17"
tags: jekyll
---


<style>
.markdown-format pre.command-line {
    border: 2px solid #ddd;
    background-color: #333;
    color: #fff;
    padding: 10px;
    padding-top: 10px;
    padding-right: 10px;
    padding-bottom: 10px;
    padding-left: 10px;
    -webkit-font-smoothing: auto;
}

.markdown-format pre.command-line .command:before {
    content: "$ ";
}
.markdown-format pre.command-line .output {
    color: #63E463;
}
pre.command-line {
}
.markdown-format pre.command-line .comment {
    color: #ccc;
}
.markdown-format pre.command-line em {
    color: #F9FE64;
    font-style: italic;
}
</style>

<div class="article-body content-body wikistyle markdown-format">
          <div class="intro">

          <p>You can set up a local version of your Jekyll GitHub Pages site to test changes to your site locally.  We highly recommend installing Jekyll to preview your site and help troubleshoot failed Jekyll builds.</p>

          </div>

          <p>In this article:</p>

<ul>
<li><a href="#requirements">Requirements</a></li>
<li><a href="#step-1-create-a-local-repository-for-your-jekyll-site">Step 1: Create a local repository for your Jekyll site</a></li>
<li><a href="#step-2-install-jekyll-using-bundler">Step 2: Install Jekyll using Bundler</a></li>
<li><a href="#step-3-optional-generate-jekyll-site-files">Step 3 (optional): Generate Jekyll site files</a></li>
<li><a href="#step-4-build-your-local-jekyll-site">Step 4: Build your local Jekyll site</a></li>
<li><a href="#keeping-your-site-up-to-date-with-the-github-pages-gem">Keeping your site up to date with the GitHub Pages gem</a></li>
</ul>

<div class="platform-windows">

<p>Jekyll is not officially supported for Windows. For more information, see "<a href="http://jekyllrb.com/docs/windows/#installation">Jekyll on Windows</a>" in the official Jekyll documentation.</p>

</div>

<h3>
<a id="requirements" class="anchor" href="#requirements" aria-hidden="true"><span class="octicon octicon-link"></span></a>Requirements</h3>

<p>We recommend using <a href="http://bundler.io/">Bundler</a> to install and run Jekyll. Bundler manages Ruby gem dependencies, reduces Jekyll build errors, and prevents environment-related bugs. To install Bundler, you must install <a href="https://www.ruby-lang.org/">Ruby</a>.</p>

<ol>
<li><p>Open <span class="platform-mac">Terminal</span><span class="platform-linux">Terminal</span><span class="platform-windows">Git Bash</span>.</p></li>
<li>
<p>Check whether you have Ruby 2.0.0 or higher installed:</p>

<pre class="command-line"><span class="command">ruby --version</span>
<span class="output">ruby <em>2.X.X</em></span>
</pre>
</li>
<li><p>If you don't have Ruby installed, <a href="https://www.ruby-lang.org/en/downloads/">install Ruby 2.0.0 or higher</a>.</p></li>
<li>
<p>Install Bundler:</p>

<pre class="command-line"><span class="command">gem install bundler</span>
<span class="comment"># Installs the Bundler gem</span>
</pre>
</li>
<li><p>If you already have a local repository for your Jekyll site, skip to <a href="#step-2-install-jekyll-using-bundler">Step 2</a>.</p></li>
</ol>

<h3>
<a id="step-1-create-a-local-repository-for-your-jekyll-site" class="anchor" href="#step-1-create-a-local-repository-for-your-jekyll-site" aria-hidden="true"><span class="octicon octicon-link"></span></a>Step 1: Create a local repository for your Jekyll site</h3>

<ol>
<li>If you haven't already downloaded Git, install it. For more information, see "<a href="/articles/set-up-git/">Set up Git</a>."</li>
<li><p>Open <span class="platform-mac">Terminal</span><span class="platform-linux">Terminal</span><span class="platform-windows">Git Bash</span>.</p></li>
<li>
<p>On your local computer, initialize a new Git repository for your Jekyll site:</p>

<pre class="command-line"><span class="command">git init <em>my-jekyll-site-project-name</em></span>
<span class="output">Initialized empty Git repository in /Users/octocat/my-site/.git/</span>
<span class="comment"># Creates a new file directory on your local computer, initialized as a Git repository</span>
</pre>
</li>
<li>
<p>Change directories to the new repository you created:</p>

<pre class="command-line"><span class="command">cd <em>my-jekyll-site-project-name</em></span>
<span class="comment"># Changes the working directory</span>
</pre>
</li>
<li>
<p>If your new local repository is for a <a href="/articles/user-organization-and-project-pages/">Project pages site</a>, create a new <code>gh-pages</code> branch:</p>

<pre class="command-line"><span class="command">git checkout -b gh-pages</span>
<span class="output">Switched to a new branch 'gh-pages'</span>
<span class="comment"># Creates a new branch called 'gh-pages', and checks it out</span>
</pre>
</li>
</ol>

<h3>
<a id="step-2-install-jekyll-using-bundler" class="anchor" href="#step-2-install-jekyll-using-bundler" aria-hidden="true"><span class="octicon octicon-link"></span></a>Step 2: Install Jekyll using Bundler</h3>

<p>To track your site's dependencies, Ruby will use the contents of your Gemfile to build your Jekyll site.</p>

<ol>
<li>
<p>Check to see if you have a Gemfile in your local Jekyll site repository:</p>

<pre class="command-line"><span class="command">ls</span>
<span class="output">Gemfile</span>
</pre>

<p>If you have a Gemfile, skip to step 4.
If you don't have a Gemfile, skip to step 2.</p>
</li>
<li>
<p>If you don't have a Gemfile, open your favorite text editor, such as <a href="https://atom.io/">Atom</a>, and add these lines to a new file:</p>

<pre class="command-line">source 'https://rubygems.org'
gem 'github-pages', group: :jekyll_plugins
</pre>
</li>
<li><p>Name the file <code>Gemfile</code> and save it to the <a href="https://en.wikipedia.org/wiki/Root_directory">root directory</a> of your local Jekyll site repository. Skip to step 5 to install Jekyll.</p></li>
<li>
<p>If you already have a Gemfile, open your favorite text editor, such as <a href="https://atom.io/">Atom</a>, and add these lines to your Gemfile:</p>

<pre class="command-line">source 'https://rubygems.org'
gem 'github-pages', group: :jekyll_plugins
</pre>
</li>
<li>
<p>Install Jekyll and other <a href="https://pages.github.com/versions/">dependencies</a> from the GitHub Pages gem:</p>

<pre class="command-line"><span class="command">bundle install</span>
<span class="output">Fetching gem metadata from https://rubygems.org/............</span>
<span class="output">Fetching version metadata from https://rubygems.org/...</span>
<span class="output">Fetching dependency metadata from https://rubygems.org/..</span>
<span class="output">Resolving dependencies...</span>
</pre>

<div class="platform-mac">

<div class="alert tip">

<p><strong>Tip:</strong> If you see a Ruby error when you try to install Jekyll using Bundler, you may need to use a package manager, such as <a href="https://rvm.io/">RVM</a> or <a href="http://brew.sh/">Homebrew</a>, to manage your Ruby installation. For more information, see <a href="http://jekyllrb.com/docs/troubleshooting/#jekyll-amp-mac-os-x-1011">Jekyll's troubleshooting page</a>.</p>

</div>

</div>
</li>
</ol>

<h3>
<a id="step-3-optional-generate-jekyll-site-files" class="anchor" href="#step-3-optional-generate-jekyll-site-files" aria-hidden="true"><span class="octicon octicon-link"></span></a>Step 3 (optional): Generate Jekyll site files</h3>

<p>To build your Jekyll site locally, preview your site changes, and troubleshoot build errors, you must have Jekyll site files on your local computer. You may already have Jekyll site files on your local computer if you cloned a Jekyll site repository. If you don't have a Jekyll site downloaded, you can generate Jekyll site files for a basic Jekyll template site in your local repository.</p>

<p>If you want to use an existing Jekyll site repository on GitHub as the starting template for your Jekyll site, fork and clone the Jekyll site repository on GitHub to your local computer. For more information, see "<a href="/articles/fork-a-repo/">Fork a repo</a>."</p>

<ol>
<li>
<p>If you don't already have a Jekyll site on your local computer, create a Jekyll template site:</p>

<pre class="command-line"><span class="command">bundle exec jekyll new . --force</span>
<span class="output">New jekyll site installed in /Users/octocat/my-site.</span>
</pre>
</li>
<li>
<p>To edit the Jekyll template site, open your new Jekyll site files in a text editor. Make your changes and save them in the text editor. You can preview these changes locally without committing your changes using Git.</p>

<p>If you want to publish your changes on your site, then you must commit your changes and push them to GitHub using Git. For more information on this workflow, see "<a href="/articles/good-resources-for-learning-git-and-github/">Good Resources for Learning Git and GitHub</a>" or see this <a href="/articles/git-cheatsheet">Git cheat sheet</a>.</p>
</li>
</ol>

<h3>
<a id="step-4-build-your-local-jekyll-site" class="anchor" href="#step-4-build-your-local-jekyll-site" aria-hidden="true"><span class="octicon octicon-link"></span></a>Step 4: Build your local Jekyll site</h3>

<ol>
<li>Navigate into the <a href="https://en.wikipedia.org/wiki/Root_directory">root directory</a> of your local Jekyll site repository.</li>
<li>
<p>Run your Jekyll site locally:</p>

<pre class="command-line"><span class="command">bundle exec jekyll serve</span>
<span class="output">Configuration file: /Users/octocat/my-site/_config.yml</span>
<span class="output">           Source: /Users/octocat/my-site</span>
<span class="output">      Destination: /Users/octocat/my-site/_site</span>
<span class="output">Incremental build: disabled. Enable with --incremental</span>
<span class="output">     Generating...</span>
<span class="output">                   done in 0.309 seconds.</span>
<span class="output">Auto-regeneration: enabled for '/Users/octocat/my-site'</span>
<span class="output">Configuration file: /Users/octocat/my-site/_config.yml</span>
<span class="output">   Server address: http://127.0.0.1:4000/</span>
<span class="output"> Server running... press ctrl-c to stop.</span>
</pre>
</li>
<li><p>Preview your local Jekyll site in your web browser at <code>http://localhost:4000</code>.</p></li>
</ol>

<h3>
<a id="keeping-your-site-up-to-date-with-the-github-pages-gem" class="anchor" href="#keeping-your-site-up-to-date-with-the-github-pages-gem" aria-hidden="true"><span class="octicon octicon-link"></span></a>Keeping your site up to date with the GitHub Pages gem</h3>

<p>Jekyll is an <a href="https://github.com/jekyll/jekyll">active open source project</a> and is updated frequently. As the GitHub Pages server is updated, the software on your computer may become out of date, resulting in your site appearing different locally from how it looks when published on GitHub.</p>

<ol>
<li><p>Open <span class="platform-mac">Terminal</span><span class="platform-linux">Terminal</span><span class="platform-windows">Git Bash</span>.</p></li>
<li>
<p>Run this update command:</p>

<ul>
<li>If you followed our setup recommendations and installed <a href="http://bundler.io/">Bundler</a>, run <code>bundle update github-pages</code> or simply <code>bundle update</code> and all your gems will update to the latest versions.</li>
<li>If you don't have Bundler installed, run <code>gem update github-pages</code>
</li>
</ul>
</li>
</ol>
