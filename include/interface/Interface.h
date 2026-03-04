/* Pablo Diez-Silva 2026- . The libPoisson interface.
 * Inspired by the libMobility interface.
 * Every solver must inherit from the Solver virtual base class.
 * See solvers/ directory for examples of solvers.
 */

#ifndef INTERFACE_H
#define INTERFACE_H
#include "defines.h"
#include "memory/container.h"
#include <vector>

namespace libPoisson {
    enum class periodicity_mode{
        single_wall,
        two_walls,
        open,
        periodic,
        uspecified
    };

    //Common parameters for all solvers.
    struct Parameters{
        std::vector<real> chargeRadius; //Radius of the charge distribution.
        real permittivity; //Permittivity of the medium.
                           //libMobility's Lanczos parameters are not needed for libPoisson, so they are not included here.
        };

    /** @brief List of parameters that cannot be changed by reinitializing a solver and/or
     * are properties of the solver.
     *
     * For instance, an open boundary solver will only
     * accept open periodicity. Another solver might be set up for either cpu or gpu
     * at creation
     */
    struct Configuration{
        periodicity_mode periodicityX = periodicity_mode::uspecified;
        periodicity_mode periodicityY = periodicity_mode::uspecified;
        periodicity_mode periodicityZ = periodicity_mode::uspecified;
    };

    class Solver{
        private:
            bool initialized = false;

        protected:
        // Protected constructor to prevent direct instantiation of the base class.
            Solver() {};

        public:
        #if defined SINGLE_PRECISION
            static constexpr auto precision = "float";
        #elif defined DOUBLE_PRECISION
            static constexpr auto precision = "double";
        #endif
        // The constructor should accept a Configuration object and ensure the
        // requested parameters are acceptable (an open boundary solver should
        // complain if periodicity is selected). A runtime_exception should be thrown
        // if the configuration is invalid. The constructor here is just an example,
        // since this is a pure virtual class
        /*
        Solver(Configuration conf){
          if(conf.periodicityX != periodicity::open or
          conf.periodicityY != periodicity::open or
          conf.periodicityZ != periodicity::open)
            throw std::runtime_error("[Mobility] This is an open boundary solver");
        }
        */
        // Outside of the common interface, solvers can define a function called
        // setParameters[ModuleName] , with arbitrary input, that simply acknowledges
        // a set of values proper to the specific solver. These new parameters should
        // NOT take effect until initialize is called afterwards.
        //  void setParametersModuleName(MyParameters par){
        //    //Store required parameters
        //  }

        // Initialize should leave the solver in a state ready for setPositions to be
        //  called. Furthermore, initialize can be called again if some parameter
        //  changes

        virtual void initialize(Parameters par){
            if (initialized) {
                this->clean();
            }
            this->initialized = true;
            // libMobility do not store hydrodynamicRadius here, do not know why. ASK RAUL.
            // this initialize function might not be needed for libPoisson, since the parameters are not expected to change, but it is included here for consistency with libMobility.
        }

        /** @brief Clean up any resources allocated by the solver. This function should be called before reinitializing the solver with new parameters or before destroying the solver instance.
         */
        virtual void clean() = 0;

        /** @brief Set the positions of the charges in the system.
         *
         * This function should be called after initialize and before any computation of the potential or field.
         * The input is a device_span of real numbers representing the positions of the charges in a flattened array (e.g., x1, y1, z1, x2, y2, z2, ...).
         */
        virtual void setPositions(device_span<const real> positions) = 0;

        virtual void getNumberParticles() = 0;

        /** @brief Compute the potential at the positions of the charges due to the other charges in the system.
         *
         * This function should be called after setPositions. The input is a device_span of real numbers
         * representing the charges of the particles and a device_span to store the computed potential at each charge's position.
         */
        virtual void computePotential(device_span<const real> charges, device_span<const real> potential) = 0;

        /** @brief Compute the electric field at the positions of the charges due to the other charges in the system.
         *
         * This function should be called after setPositions. The input is a device_span of real numbers
         * representing the charges of the particles and a device_span to store the computed electric
         * field at each charge's position (flattened array: Ex1, Ey1, Ez1, Ex2, Ey2, Ez2, ...).
         */
        virtual void computeField(device_span<const real> charges, device_span<const real> field) = 0;

        /** @brief Compute the force on each charge due to the other charges in the system.
         *
         * This function should be called after setPositions. The input is a device_span of real
         * numbers representing the charges of the particles and a device_span to store the computed
         * force on each charge (flattened array: Fx1, Fy1, Fz1, Fx2, Fy2, Fz2, ...).
         * Is separated from computeField because some solvers might take into account the size of the
         * charge distribution, and thus the force is not simply the field times the charge.
         */
        virtual void computeForce(device_span<const real> charges, device_span<const real> force) = 0;

        /** @brief Compute the energy of each charge in the system.
         *
         * This function should be called after setPositions. The input is a device_span of real
         * numbers representing the charges of the particles and a device_span to store the
         * computed energy of the system. Is separated from computePotential because some solvers
         * might take into account the size of the charge distribution, and thus the energy is not simply the potential times the charge.
         */
        virtual void computeEnergy(device_span<const real> charges, device_span<const real> energy) = 0;
    }; // class Solver
} // namespace libPoisson
#endif // INTERFACE_H
