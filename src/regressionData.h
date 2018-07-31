#ifndef __REGRESSIONDATA_HPP__
#define __REGRESSIONDATA_HPP__

#include "fdaPDE.h"
#include "mesh_objects.h"
#include "param_functors.h"

//!  An IO handler class for objects passed from R
/*!
 * This class, given the data from R, convert them in a C++ format, offering a
 * series of method for their access, so isolating the more the possible the specific
 * code for R/C++ data conversion.
*/
class  RegressionData{
	private:

		// Design matrix pointer and dimensions
		std::vector<Point> locations_;

		VectorXr observations_;
		std::vector<UInt> observations_indices_;
		bool locations_by_nodes_;


		//Design matrix
		MatrixXr covariates_;
		UInt n_;
		UInt p_;

		//Other parameters
		UInt order_;
		std::vector<Real> lambda_;
		UInt GCVmethod_;
		UInt nrealizations_;      // Number of relizations for the stochastic estimation of GCV

		std::vector<Real> bc_values_;
		std::vector<UInt> bc_indices_;

		//bool inputType;
		bool DOF_;

		#ifdef R_VERSION_
		void setLocations(SEXP Rlocations);
		void setObservations(SEXP Robservations);
		void setCovariates(SEXP Rcovariates);
		void setNrealizations(SEXP Rnrealizations);
		#endif

	public:
		//! A basic version of the constructor.

		/*!
			It initializes the object storing the R given objects. This is the simplest of the two possible interfaces with R
			\param Robservations an R-vector containing the values of the observations.
			\param Rdesmat an R-matrix containing the design matrix for the regression.
			\param Rorder an R-integer containing the order of the approximating basis.
			\param Rlambda an R-double containing the penalization term of the empirical evidence respect to the prior one.
			\param Rbindex an R-integer containing the indexes of the nodes the user want to apply a Dirichlet Condition,
					the other are automatically considered in Neumann Condition.
			\param Rbvalues an R-double containing the value to impose for the Dirichlet condition, on the indexes specified in Rbindex
		*/


		//! A complete version of the constructor.
		/*!
			It initializes the object storing the R given objects. This is the simplest of the two possible interfaces with R
			\param Robservations an R-vector containing the values of the observations.
			\param Rdesmat an R-matrix containing the design matrix for the regression.
			\param Rorder an R-integer containing the order of the approximating basis.
			\param Rlambda an R-double containing the penalization term of the empirical evidence respect to the prior one.
			\param Rbindex an R-integer vector containing the indexes of the nodes the user want to apply a Dirichlet Condition,
					the other are automatically considered in Neumann Condition.
			\param Rbvalues an R-double vector containing the value to impose for the Dirichlet condition, on the indexes specified in Rbindex
			\param Rc an R-double that contains the coefficient of the REACTION term
			\param Rbeta an R-double 2-dim vector that contains the coefficients for the TRANSPORT coefficients.
			\param RK an R-double 2X2 matrix containing the coefficients for a anisotropic DIFFUSION term.
			\param (UNSUPPORTED put it zero) Ru an R-double vector of length #triangles that contaiins the forcing term integrals.
		*/
		RegressionData(){};

		#ifdef R_VERSION_
		explicit RegressionData(SEXP Rlocations, SEXP Robservations, SEXP Rorder, SEXP Rlambda, SEXP Rcovariates,
				   SEXP RBCIndices, SEXP RBCValues, SEXP DOF,SEXP RGCVmethod, SEXP Rnrealizations);
		#endif

		explicit RegressionData(const std::vector<Point>& locations, const VectorXr& observations, UInt order, const std::vector<Real>& lambda, const MatrixXr& covariates , const std::vector<UInt>& bc_indices, const std::vector<Real>& bc_values, bool DOF, UInt GCVmethod, UInt nrealizations);


		void printObservations(std::ostream & out) const;
		void printCovariates(std::ostream & out) const;
		void printLocations(std::ostream & out) const;

		//! A method returning a reference to the observations vector
		inline VectorXr const & getObservations() const {return observations_;}
		//! A method returning a reference to the design matrix
		inline MatrixXr const & getCovariates() const {return covariates_;}
		//! A method returning the number of observations
		inline UInt const getNumberofObservations() const {return observations_.size();}
		//! A method returning the locations of the observations
		inline std::vector<Point> const & getLocations() const {return locations_;}
		inline bool isLocationsByNodes() const {return locations_by_nodes_;}
		inline bool computeDOF() const {return DOF_;}
		inline std::vector<UInt> const & getObservationsIndices() const {return observations_indices_;}
		//! A method returning the the penalization term
		inline std::vector<Real> const & getLambda() const {return lambda_;}
		//! A method returning the input order
		inline UInt const getOrder() const {return order_;}
		//! A method returning the indexes of the nodes for which is needed to apply Dirichlet Conditions
		inline std::vector<UInt> const & getDirichletIndices() const {return bc_indices_;}
		//! A method returning the values to apply for Dirichlet Conditions
		inline std::vector<Real> const & getDirichletValues() const {return bc_values_;}
		//! A method returning the method that should be used to compute the GCV:
		//! 1: exact calculation
		//! 2: stochastic estimation
		inline UInt const & getGCVmethod() const {return GCVmethod_;}
		//! A method returning the number of vectors to use to stochastically estimate the edf
		inline UInt const & getNrealizations() const {return nrealizations_;}

        // A method to set the DOF computation
        inline void setComputeDOF(const bool dof) {DOF_ = dof;}
        // A method to set the penalization terms
        inline void setLambda(const std::vector<Real>& lambda) {lambda_ = lambda;}
};


class  RegressionDataElliptic:public RegressionData
{
	private:
		Eigen::Matrix<Real,2,2> K_;
		Eigen::Matrix<Real,2,1> beta_;
		Real c_;

	public:
		#ifdef R_VERSION_
		explicit RegressionDataElliptic(SEXP Rlocations, SEXP Robservations, SEXP Rorder, SEXP Rlambda, SEXP RK, SEXP Rbeta,
				 SEXP Rc, SEXP Rcovariates, SEXP RBCIndices, SEXP RBCValues, SEXP DOF,SEXP RGCVmethod, SEXP Rnrealizations);
		#endif

		explicit RegressionDataElliptic(const std::vector<Point>& locations, const VectorXr& observations, UInt order, const std::vector<Real>& lambda, const Eigen::Matrix<Real,2,2>& K,	const Eigen::Matrix<Real,2,1>& beta,
		Real c, const MatrixXr& covariates , const std::vector<UInt>& bc_indices, const std::vector<Real>& bc_values, bool DOF, UInt GCVmethod, UInt nrealizations);

		inline Eigen::Matrix<Real,2,2> const & getK() const {return K_;}
		inline Eigen::Matrix<Real,2,1> const & getBeta() const {return beta_;}
		inline Real const getC() const {return c_;}

        inline void setK(const Eigen::Matrix<Real, 2, 2> & K) {K_ = K;}
};

class RegressionDataEllipticSpaceVarying:public RegressionData
{
	private:
		Diffusivity K_;
		Advection beta_;
		Reaction c_;
		ForcingTerm u_;

	public:
		#ifdef R_VERSION_
		explicit RegressionDataEllipticSpaceVarying(SEXP Rlocations, SEXP Robservations, SEXP Rorder, SEXP Rlambda, SEXP RK, SEXP Rbeta,
				 SEXP Rc, SEXP Ru, SEXP Rcovariates, SEXP RBCIndices, SEXP RBCValues, SEXP DOF,SEXP RGCVmethod, SEXP Rnrealizations);
		#endif


		explicit RegressionDataEllipticSpaceVarying(std::vector<Point>& locations, VectorXr& observations, UInt order, std::vector<Real> lambda, const std::vector<Eigen::Matrix<Real,2,2>, Eigen::aligned_allocator<Eigen::Matrix<Real,2,2> > >& K,	const std::vector<Eigen::Matrix<Real,2,1>, Eigen::aligned_allocator<Eigen::Matrix<Real,2,1> > >& beta,
				const std::vector<Real>& c, const std::vector<Real>& u, MatrixXr& covariates , std::vector<UInt>& bc_indices, std::vector<Real>& bc_values, bool DOF, UInt GCVmethod, UInt nrealizations);

		inline Diffusivity const & getK() const {return K_;}
		inline Advection const & getBeta() const {return beta_;}
		inline Reaction const & getC() const {return c_;}
		inline ForcingTerm const & getU() const {return u_;}

		void print(std::ostream & out) const;
};

#include "regressionData_imp.h"

#endif
