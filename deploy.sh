npm install
rm -Rf _site
jekyll b
mv _site ../
mv CNAME ../_site/
git checkout master
rm -Rf *
rm -Rf .j*
mv ../_site/* .
git add .
