create schema if not exists kaggle_retail;

create table if not exists kaggle_retail.dim_campaigns(
	campaign_sk int,
	campaign_id varchar,
	campaign_name varchar,
	start_date_sk int,
	end_date_sk int,
	campaign_budget int);

create table if not exists kaggle_retail.dim_customers(
	customer_sk int,
	customer_id varchar,
	first_name varchar,
	last_name varchar,
	email varchar,
	residential_location varchar,
	customer_segment varchar);

create table if not exists kaggle_retail.dim_dates(
	full_date date,
	date_sk int,
	year int,
	month int,
	day int,
	weekday int,
	quarter int);

create table if not exists kaggle_retail.dim_products(
	product_sk int,
	product_id varchar,
	product_name varchar,
	category varchar,
	brand varchar,
	origin_location varchar);

create table if not exists kaggle_retail.dim_salespersons(
	salesperson_sk int,
	salesperson_id varchar,
	salesperson_name varchar,
	salesperson_role varchar);

create table if not exists kaggle_retail.dim_stores(
	store_sk int,
	store_id varchar,
	store_name varchar,
	store_type varchar,
	store_location varchar,
	store_manager_sk int);

create table if not exists kaggle_retail.fact_sales_normalized(
	sales_sk int,
	sales_id varchar,
	customer_sk int,
	product_sk int,
	store_sk int,
	salesperson_sk int,
	campaign_sk int,
	sales_date timestamp,
	total_amount real);
















