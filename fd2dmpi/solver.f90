!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!  The subroutines below are adapted from the Computational Toolkit provided in: 
!   
!      Schuster, G. T. (2017). Seismic inversion. Society of Exploration Geophysicists.
!
!  We kindly thank Prof. Schuster for allowing us to use these useful and efficient Fortran 
!  subroutines. Please Cite the book above if you use these subroutines.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


! module for 2D applications including forward and adjoint modeling
!
module solver

use fdcore
use datatype

implicit none

type(param),       private              :: par
type(acquisition), private              :: coord
logical,           private              :: store_snap, message
integer,           private              :: ix, iz, it, is, is1, is2, j,i,num_tmp,z1_tmp,z2_tmp,x1_tmp,x2_tmp
integer,           private, allocatable :: fs(:)
real,              private, allocatable :: s(:), c(:,:), den(:,:), s_adj(:,:), &
                                           p_end(:,:), u_end(:,:), w_end(:,:), boundary(:,:), &
                                           p0_bk(:,:), p_bk(:,:), u_bk(:,:), w_bk(:,:), &
                                           dg(:,:), dfw(:,:), dbk(:,:), g(:,:), fw(:,:), bk(:,:)

                                           
contains

!-----------------------------------------------------------------------------------------
subroutine forward_modeling(parfile)

  use io
  use math
  use mmi_mpi
  use pml
  use source

  character(len=*), intent(in) :: parfile
  character(len=200)      :: str
  call start_mpi

  ! Read input parameters
  call readparamfile(parfile, par) !MPI

  ! PML setting: damp & damp_global
  call init_pml(par%nx, par%nz, par%npml)

  ! Read acquisition geometry data
  call readcoordfile(par%coordfile, coord) !MPI

  ! Memory allocations
  allocate(s(par%nt))
  allocate(fs(nx_pml))
  allocate(c(nz_pml,nx_pml), den(nz_pml,nx_pml))

  ! Set up free surface
  call free_surface(par, fs, npml) !MPI

  ! Read velocity model
  call readvelfile(par,c,npml,nx_pml,nz_pml) !MPI

  ! Read density model
  call read_densityfile(par,c,den,npml,nx_pml,nz_pml) !MPI

  if (rank==0) then
    write(*,*) '************************************************************'
    write(*,*) ' '
    write(*,*) ' Forward modeling ---- time-domain staggered-42-FD          '
    write(*,*) ' '
    write(*,*) '************************************************************'
  endif

  ! Forward modeling using dynamic load balancing
  message = .true.
  store_snap = .false.
  if (par%store_snap .eq. 1) store_snap = .true.

  call get_assigned(1, coord%ns, is1, is2)

  ! Read g num
  ! allocate(item(coord%ns, 2))
  ! open(unit = 11, FILE = './model/record_num/rec_num.dat')
  ! do i = 1,coord%ns,1
  !   READ(11,*)(item(i,j),j=1,2)
  ! enddo
  ! close(11)

  do is = is1, is2, 1
    call filename(str,'./parfile/forward_source/src',is,'.bin')
    call read_binfile(str, s, par%nt)

    if (message) then
      write(*,*) 'Process ', rank, ', shot', is, 'source max value = ', maxval(s)
      flush(6)
    endif

    ! par%nx = int( item(is, 2) - item(is, 1) + 1 )
    ! coord%ngmax = int( item(is, 2) - item(is, 1) + 1 )
    ! coord%xs(is) = coord%xs(is) - (item(is,1) - 1) * par%dx
    ! coord%ng(is) = int( item(is, 2) - item(is, 1) + 1 )
    ! coord%xg(is,:) = coord%xg(is,:) - (item(is,1) - 1) *par%dx
    deallocate(damp)
    deallocate(damp_global)
   ! PML setting: damp & damp_global
    call init_pml(par%nx, par%nz, par%npml)
    ! allocate(c1(nz_pml,nx_pml), den1(nz_pml,nx_pml))
    ! allocate(fs1(nx_pml))
    ! den1(:, npml+1:npml+par%nx ) = den(:, int(npml+item(is,1)):int(npml+item(is,2)))
    ! do j =1, npml, 1
    !   den1(:,j) = den1(:, npml+1)
    !   den1(:,nx_pml + 1-j) = den1(:, npml+par%nx) 
    ! enddo
    ! c1(:, npml+1:npml+par%nx ) = c(:, int(npml+item(is,1)):int(npml+item(is,2)))
    ! do j =1, npml, 1
    !   c1(:,j) = c1(:, npml+1)
    !   c1(:,nx_pml + 1-j) = c1(:, npml+par%nx) 
    ! enddo
    ! fs1(npml+1:npml+par%nx) = fs(par%npml+item(is,1):par%npml+item(is,2))
    ! fs1(1:npml) = fs1(npml+1)
    ! fs1(nx_pml-npml+1:nx_pml) = fs1(nx_pml-npml)
    par%vmin = min_value(c(iz1:iz2,ix1:ix2),par%nz,par%nx,fs-npml)
    call setup_pml(par%dx, par%vmin)
   
    !par%fileformat = 'bi' 
    
    call staggered42_modeling_is(is, par, coord, s, c, den, fs, nx_pml, nz_pml, npml, damp_global, store_snap)
    ! deallocate(c, fs, den)
  enddo


  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!   
  call MPI_Barrier(MPI_COMM_WORLD, ierr)

  ! deallocate(c, den, s, fs, damp, damp_global, item)
  deallocate(c, den, s, fs, damp, damp_global)

  999 continue

  call stop_mpi

end subroutine forward_modeling

!-----------------------------------------------------------------------------------------
subroutine adjoint_modeling(parfile)

  use io
  use math
  use mmi_mpi
  use pml
  use source
  
  character(len=*), intent(in) :: parfile
  character(len=200)      :: str

   
  call start_mpi
  
  ! Read input parameters
  call readparamfile(parfile, par) !MPI
  
  !record whole model nx
  num_tmp = par%nx
  
  ! PML setting: damp & damp_global
  call init_pml(par%nx, par%nz, par%npml) 
  ! z1_tmp = iz1
  ! z2_tmp = iz2
  ! x1_tmp = ix1
  ! x2_tmp = ix2

  ! Read acquisition geometry data
  call readcoordfile(par%coordfile, coord) !MPI

  ! Memory allocations
  allocate(s(par%nt))
  allocate(fs(nx_pml))
  allocate(c(nz_pml,nx_pml), den(nz_pml,nx_pml))

  allocate(dg(par%nz,par%nx))
  allocate(dfw(par%nz,par%nx))
  allocate(dbk(par%nz,par%nx))
  allocate(g(par%nz,par%nx))  
  allocate(fw(par%nz,par%nx))
  allocate(bk(par%nz,par%nx))

  ! Set up free surface
  call free_surface(par, fs, npml) !MPI

  ! Read velocity model
  call readvelfile(par,c,npml,nx_pml,nz_pml) !MPI
  
  ! Read density model
  call read_densityfile(par,c,den,npml,nx_pml,nz_pml) !MPI
  
   
  if (rank==0) then
    write(*,*) '************************************************************'
    write(*,*) ' '
    write(*,*) ' Adjoint modeling ---- time-domain staggered-42-FD          '
    write(*,*) ' '
    write(*,*) '************************************************************'
  endif
  ! Adjoint modeling using dynamic load balancing
  message = .true.
  store_snap = .false.
  if (par%store_snap .eq. 1) store_snap = .true.

  call get_assigned(1, coord%ns, is1, is2)

  ! Initialize
  dg   = 0.0
  dfw  = 0.0
  dbk  = 0.0


  ! Read rec number on grid 

  ! allocate(item1(coord%ns, 2))
  ! open(unit = 12, FILE = './model/record_num/rec_num.dat')
  ! do i = 1,coord%ns,1
  !   READ(12,*)(item1(i,j),j=1,2)
  ! enddo
  ! close(12)
  do is = is1, is2, 1
    
    
    ! Read forward source 
    call filename(str,'./parfile/forward_source/src',is,'.bin')
    call read_binfile(str, s, par%nt)

    ! par%nx = int( item1(is, 2) - item1(is, 1) + 1 )
    ! coord%ngmax = int( item1(is, 2) - item1(is, 1) + 1 )

    ! Read adjoint source
    allocate(s_adj(par%nt, coord%ngmax))
    call filename(str,'./parfile/adjoint_source/src',is,'.bin')
    call read_binfile(str, s_adj, par%nt, coord%ngmax) 

    ! Read boundary, p_end, u_end, w_end
    allocate(boundary(par%nx*9+par%nz*6,par%nt))
    allocate(p_end(par%nz,par%nx))
    allocate(u_end(par%nz,par%nx))
    allocate(w_end(par%nz,par%nx))
    call filename(str, par%data_out, is, '_snapshot/boundary.bin')
    call read_binfile(str, boundary, par%nx*9+par%nz*6, par%nt)
    call filename(str, par%data_out, is, '_snapshot/p_end.bin')
    call read_binfile(str, p_end, par%nz, par%nx)
    call filename(str, par%data_out, is, '_snapshot/u_end.bin')
    call read_binfile(str, u_end, par%nz, par%nx)
    call filename(str, par%data_out, is, '_snapshot/w_end.bin')
    call read_binfile(str, w_end, par%nz, par%nx)

    ! Initial parameter
    ! coord%xs(is) = coord%xs(is) - (item1(is,1) - 1) * par%dx
    ! coord%ng(is) = int( item1(is, 2) - item1(is, 1) + 1 )
    ! coord%xg(is,:) = coord%xg(is,:) - (item1(is,1) - 1) *par%dx
    deallocate(damp)
    deallocate(damp_global)
    call init_pml(par%nx, par%nz, par%npml)
    ! allocate(c2(nz_pml,nx_pml), den2(nz_pml,nx_pml))
    ! allocate(fs2(nx_pml))
    allocate(p0_bk(nz_pml,nx_pml))
    allocate(p_bk(nz_pml,nx_pml))
    allocate(u_bk(nz_pml,nx_pml))
    allocate(w_bk(nz_pml,nx_pml))

    p_bk = 0.0
    u_bk = 0.0
    w_bk = 0.0
    
    ! den2(:, npml+1:npml+par%nx ) = den(:, int(npml+item1(is,1)):int(npml+item1(is,2)))
    ! do j =1, npml, 1
    !   den2(:,j) = den2(:, npml+1)
    !   den2(:,nx_pml + 1-j) = den2(:, npml+par%nx) 
    ! enddo
    ! c2(:, npml+1:npml+par%nx ) = c(:, int(npml+item1(is,1)):int(npml+item1(is,2)))
    ! do j =1, npml, 1
    !   c2(:,j) = c2(:, npml+1)
    !   c2(:,nx_pml + 1 - j ) = c2(:, npml+par%nx)
    ! enddo
    ! fs2(npml+1:npml+par%nx) = fs(par%npml+item1(is,1):par%npml+item1(is,2))
    ! fs2(1:npml) = fs2(npml+1)
    ! fs2(nx_pml-npml+1:nx_pml) = fs2(nx_pml-npml)

    ! Determine minimum velocity
    par%vmin = min_value(c(iz1:iz2,ix1:ix2),par%nz,par%nx,fs-npml)

    ! Setup PML damping coefficient
    call setup_pml(par%dx, par%vmin)

    ! Decide p,u,w_new

    do it=par%nt,2,-1

      p0_bk = p_bk
      
      call staggered42_reco_it(is, it, par, coord, s, c(iz1:iz2,ix1:ix2), den(iz1:iz2,ix1:ix2), fs, &
                                boundary, p_end, u_end, w_end)
      call staggered42_back_it(is, it, par, coord, s_adj, c, den, fs, nx_pml, nz_pml, npml, damp_global, p_bk, u_bk,w_bk)
      dg  = dg  + p_end * (p0_bk(iz1:iz2,ix1:ix2)-p_bk(iz1:iz2,ix1:ix2))
      dfw = dfw + p_end * p_end
      dbk = dbk + p_bk(iz1:iz2,ix1:ix2) *  p_bk(iz1:iz2,ix1:ix2)
      ! dg  = dg(:,int(item1(is,1)):int(item1(is,2))) + &
      !                                            p_end  * (p0_bk(iz1:iz2,ix1:ix2) - p_bk(iz1:iz2,ix1:ix2))

      ! dfw(:,int(item1(is,1)):int(item1(is,2)))  = dfw(:,int(item1(is,1)):int(item1(is,2))) + p_end * p_end

      ! dbk(:,int(item1(is,1)):int(item1(is,2)))  = dbk(:,int(item1(is,1)):int(item1(is,2))) + &
      !                                             p_bk(iz1:iz2,ix1:ix2) * p_bk(iz1:iz2,ix1:ix2)

    enddo
    deallocate(s_adj, boundary, p_end, u_end, w_end,p0_bk, p_bk, u_bk, w_bk)
  enddo


  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!   
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(dg,   g, par%nz*num_tmp, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(dfw, fw, par%nz*num_tmp, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(dbk, bk, par%nz*num_tmp, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  

  if (rank == 0) then
    
    ! perform the proper scale to the gradient 
    g = 2.0 * g / c(iz1:iz2,ix1:ix2)

    call filename(output, par%data_out, 0, '_kernel_vp.bin')
    call write_binfile(output,  g, par%nz, num_tmp) 

    call filename(output, par%data_out, 0, '_illum_forw.bin')
    call write_binfile(output, fw, par%nz, num_tmp) 

    call filename(output, par%data_out, 0, '_illum_back.bin')
    call write_binfile(output, bk, par%nz, num_tmp) 

  endif

  ! call write_binfile('/data/xuejing/SWIT-1.0/examples/fw.bin',fw,par%nz,par%nx)


  ! deallocate(c, den, s, fs, damp, damp_global,item1)
  deallocate(c, den, s, fs, damp, damp_global)
  deallocate(dg, dfw, dbk, g, fw, bk)

  999 continue
  call stop_mpi

  end subroutine adjoint_modeling
  

end module solver
