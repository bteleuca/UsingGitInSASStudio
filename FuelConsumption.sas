/* Add macros */
%let report_column=MPG_City;

/* Determine the most fuel-efficient cars for each Origin */
proc sql;
   create table most_efficient as
   select Origin, Make, Model, MPG_City
   from sashelp.cars
   group by Origin
   having MPG_City = max(MPG_City);
quit;

/* Generate the report for the most fuel-efficient cars */
proc print data=most_efficient noobs;
   by Origin;
   var Make Model MPG_City;
   title "Most Fuel-Efficient Cars by Origin";
run;
